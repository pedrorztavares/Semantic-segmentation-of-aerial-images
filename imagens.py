import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from patchify import patchify
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    concatenate,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path("treino")

IMAGES_DIR = BASE_DIR / "imagens"
MASKS_DIR = BASE_DIR / "mascaras"

MODELS_DIR = Path("modelos")
MODELS_DIR.mkdir(exist_ok=True)

PATCH_SIZE = 256

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ============================================================
# CLASSES DA SEGMENTAÇÃO
# ============================================================

CLASS_COLORS = {
    "background": (0, 0, 0),
    "campo_agricola": (210, 210, 24),
    "floresta": (36, 179, 83),
    "agua": (45, 12, 212),
    "estrada": (255, 255, 255),
    "edificio": (245, 147, 49),
}

CLASS_NAMES = list(CLASS_COLORS.keys())
NUMBER_OF_CLASSES = len(CLASS_NAMES)

# Colormap fixo construído a partir de CLASS_COLORS.
# A ordem é a mesma de CLASS_NAMES, por isso o id de cada classe
# (0, 1, 2, ...) corresponde sempre à mesma cor, tanto nas máscaras
# originais como nas previsões do modelo. Nada de cores "aleatórias"
# escolhidas automaticamente pelo matplotlib (ex: "tab10").
SEGMENTATION_CMAP = ListedColormap(
    [tuple(channel / 255.0 for channel in CLASS_COLORS[name]) for name in CLASS_NAMES]
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def list_files(folder: Path, valid_extensions: set[str]) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"A pasta não existe: {folder}")

    files = [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in valid_extensions
    ]

    return sorted(files)


def create_file_map(files: list[Path]) -> dict[str, Path]:
    return {file.stem: file for file in files}


def find_image_mask_pairs() -> list[tuple[Path, Path]]:
    image_files = list_files(IMAGES_DIR, IMAGE_EXTENSIONS)
    mask_files = list_files(MASKS_DIR, MASK_EXTENSIONS)

    image_map = create_file_map(image_files)
    mask_map = create_file_map(mask_files)

    common_names = sorted(set(image_map) & set(mask_map))

    images_without_mask = sorted(set(image_map) - set(mask_map))
    masks_without_image = sorted(set(mask_map) - set(image_map))

    if images_without_mask:
        print("\nImagens sem máscara correspondente:")
        for name in images_without_mask:
            print(f"  - {image_map[name].name}")

    if masks_without_image:
        print("\nMáscaras sem imagem correspondente:")
        for name in masks_without_image:
            print(f"  - {mask_map[name].name}")

    if not common_names:
        raise RuntimeError(
            "Não foi encontrado nenhum par imagem/máscara.\n"
            "Confirma se os ficheiros têm o mesmo nome-base."
        )

    return [(image_map[name], mask_map[name]) for name in common_names]


def load_rgb_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_rgb_mask(mask_path: Path) -> np.ndarray:
    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)

    if mask_bgr is None:
        raise ValueError(f"Não foi possível abrir a máscara: {mask_path}")

    return cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)


def crop_to_patch_size(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "A imagem e a máscara têm dimensões diferentes: "
            f"imagem={image.shape[:2]}, máscara={mask.shape[:2]}"
        )

    height, width = image.shape[:2]

    cropped_height = (height // patch_size) * patch_size
    cropped_width = (width // patch_size) * patch_size

    if cropped_height == 0 or cropped_width == 0:
        raise ValueError(
            f"A imagem tem dimensões inferiores a {patch_size}x{patch_size}: "
            f"{width}x{height}"
        )

    cropped_image = image[:cropped_height, :cropped_width]
    cropped_mask = mask[:cropped_height, :cropped_width]

    return cropped_image, cropped_mask


def rgb_mask_to_class_mask(mask_rgb: np.ndarray) -> np.ndarray:
    class_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    recognized_pixels = np.zeros(mask_rgb.shape[:2], dtype=bool)

    for class_id, class_name in enumerate(CLASS_NAMES):
        class_color = np.array(CLASS_COLORS[class_name], dtype=np.uint8)

        pixels_of_class = np.all(mask_rgb == class_color, axis=-1)

        class_mask[pixels_of_class] = class_id
        recognized_pixels |= pixels_of_class

    unrecognized_count = np.count_nonzero(~recognized_pixels)

    if unrecognized_count > 0:
        unique_unknown_colors = np.unique(
            mask_rgb[~recognized_pixels].reshape(-1, 3),
            axis=0,
        )

        print(
            f"\nAviso: foram encontrados {unrecognized_count} píxeis "
            "com cores não definidas."
        )

        print("Algumas cores RGB não reconhecidas:")

        for color in unique_unknown_colors[:10]:
            print(f"  - {tuple(int(value) for value in color)}")

    return class_mask


def create_patches(
    image: np.ndarray,
    class_mask: np.ndarray,
    patch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    image_patches = patchify(
        image,
        (patch_size, patch_size, 3),
        step=patch_size,
    )

    mask_patches = patchify(
        class_mask,
        (patch_size, patch_size),
        step=patch_size,
    )

    image_patch_list = []
    mask_patch_list = []

    number_of_rows = image_patches.shape[0]
    number_of_columns = image_patches.shape[1]

    for row in range(number_of_rows):
        for column in range(number_of_columns):
            image_patch = image_patches[row, column, 0]
            mask_patch = mask_patches[row, column]

            image_patch = image_patch.astype(np.float32) / 255.0
            mask_patch = mask_patch.astype(np.uint8)

            image_patch_list.append(image_patch)
            mask_patch_list.append(mask_patch)

    return image_patch_list, mask_patch_list


def split_image_mask_pairs(
    pairs,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
):
    if not np.isclose(
        train_ratio + validation_ratio + test_ratio,
        1.0,
    ):
        raise ValueError(
            "As percentagens de treino, validação e teste devem somar 1."
        )

    train_pairs, temporary_pairs = train_test_split(
        pairs,
        test_size=validation_ratio + test_ratio,
        random_state=random_state,
        shuffle=True,
    )

    test_fraction_of_temporary = (
        test_ratio / (validation_ratio + test_ratio)
    )

    validation_pairs, test_pairs = train_test_split(
        temporary_pairs,
        test_size=test_fraction_of_temporary,
        random_state=random_state,
        shuffle=True,
    )

    print("\nDivisão do dataset:")
    print(f"Treino: {len(train_pairs)} imagens")
    print(f"Validação: {len(validation_pairs)} imagens")
    print(f"Teste: {len(test_pairs)} imagens")

    return train_pairs, validation_pairs, test_pairs


def prepare_dataset_from_pairs(pairs, dataset_name):
    image_dataset = []
    mask_dataset = []

    print(f"\nA preparar o conjunto de {dataset_name}...")

    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        print(
            f"[{index}/{len(pairs)}] "
            f"{image_path.name} | {mask_path.name}"
        )

        image = load_rgb_image(image_path)
        mask_rgb = load_rgb_mask(mask_path)

        image, mask_rgb = crop_to_patch_size(
            image,
            mask_rgb,
            PATCH_SIZE,
        )

        class_mask = rgb_mask_to_class_mask(mask_rgb)

        image_patches, mask_patches = create_patches(
            image,
            class_mask,
            PATCH_SIZE,
        )

        image_dataset.extend(image_patches)
        mask_dataset.extend(mask_patches)

    images_array = np.asarray(image_dataset, dtype=np.float32)
    masks_array = np.asarray(mask_dataset, dtype=np.uint8)

    print(f"{dataset_name}: {len(images_array)} patches criados.")
    print(f"Formato das imagens: {images_array.shape}")
    print(f"Formato das máscaras: {masks_array.shape}")
    print(f"Classes encontradas: {np.unique(masks_array)}")

    return images_array, masks_array


def add_class_legend():
    """Adiciona uma legenda com o nome e a cor de cada classe ao gráfico atual."""
    legend_handles = [
        Patch(
            facecolor=tuple(channel / 255.0 for channel in CLASS_COLORS[name]),
            edgecolor="black",
            label=name,
        )
        for name in CLASS_NAMES
    ]

    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize="small",
    )


def show_random_examples(images, masks, number_of_examples=15):
    number_of_examples = min(number_of_examples, len(images))

    selected_indexes = random.sample(
        range(len(images)),
        number_of_examples,
    )

    for index in selected_indexes:
        plt.figure(figsize=(11, 5))

        plt.subplot(1, 2, 1)
        plt.title(f"Imagem - patch {index}")
        plt.imshow(images[index])
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Máscara - patch {index}")
        plt.imshow(
            masks[index],
            cmap=SEGMENTATION_CMAP,
            vmin=0,
            vmax=NUMBER_OF_CLASSES - 1,
        )
        plt.axis("off")
        add_class_legend()

        plt.tight_layout()
        plt.show()


# ============================================================
# MÉTRICA JACCARD / IOU
# ============================================================

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (intersection + 1.0) / (
        K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0
    )


# ============================================================
# MODELO U-NET
# ============================================================

def build_unet_model(
    image_height: int,
    image_width: int,
    image_channels: int,
    number_of_classes: int,
) -> Model:
    inputs = Input((image_height, image_width, image_channels))

    # Encoder
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation="relu", padding="same")(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation="relu", padding="same")(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation="relu", padding="same")(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation="relu", padding="same")(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(c5)

    # Decoder
    u6 = Conv2DTranspose(
        128,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(c5)

    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(c6)

    u7 = Conv2DTranspose(
        64,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(c6)

    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(c7)

    u8 = Conv2DTranspose(
        32,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(c7)

    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation="relu", padding="same")(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation="relu", padding="same")(c8)

    u9 = Conv2DTranspose(
        16,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(c8)

    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation="relu", padding="same")(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation="relu", padding="same")(c9)

    outputs = Conv2D(
        number_of_classes,
        (1, 1),
        activation="softmax",
    )(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def train_unet_model(
    train_images,
    train_masks,
    validation_images,
    validation_masks,
    epochs=60,
    batch_size=4,
):
    train_masks_cat = to_categorical(
        train_masks,
        num_classes=NUMBER_OF_CLASSES,
    )

    validation_masks_cat = to_categorical(
        validation_masks,
        num_classes=NUMBER_OF_CLASSES,
    )

    image_height = train_images.shape[1]
    image_width = train_images.shape[2]
    image_channels = train_images.shape[3]

    model = build_unet_model(
        image_height=image_height,
        image_width=image_width,
        image_channels=image_channels,
        number_of_classes=NUMBER_OF_CLASSES,
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", jacard_coef],
    )

    model.summary()

    checkpoint = ModelCheckpoint(
        filepath=MODELS_DIR / "melhor_modelo_unet.keras",
        monitor="val_jacard_coef",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.fit(
        train_images,
        train_masks_cat,
        validation_data=(validation_images, validation_masks_cat),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint],
    )

    return model, history


# ============================================================
# GRÁFICOS E AVALIAÇÃO
# ============================================================

def plot_training_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Loss treino")
    plt.plot(history.history["val_loss"], label="Loss validação")
    plt.title("Training and validation loss")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Accuracy treino")
    plt.plot(history.history["val_accuracy"], label="Accuracy validação")
    plt.title("Training and validation accuracy")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["jacard_coef"], label="IoU/Jaccard treino")
    plt.plot(
        history.history["val_jacard_coef"],
        label="IoU/Jaccard validação",
    )
    plt.title("Training and validation IoU/Jaccard")
    plt.xlabel("Épocas")
    plt.ylabel("IoU/Jaccard")
    plt.legend()
    plt.show()


def save_training_results(history):
    final_train_accuracy = history.history["accuracy"][-1]
    final_val_accuracy = history.history["val_accuracy"][-1]
    final_train_iou = history.history["jacard_coef"][-1]
    final_val_iou = history.history["val_jacard_coef"][-1]

    best_val_accuracy = max(history.history["val_accuracy"])
    best_val_iou = max(history.history["val_jacard_coef"])

    best_val_accuracy_epoch = (
        history.history["val_accuracy"].index(best_val_accuracy) + 1
    )

    best_val_iou_epoch = (
        history.history["val_jacard_coef"].index(best_val_iou) + 1
    )

    print("\n===== RESULTADOS FINAIS =====")
    print(f"Accuracy treino final: {final_train_accuracy * 100:.2f}%")
    print(f"Accuracy validação final: {final_val_accuracy * 100:.2f}%")
    print(f"IoU/Jaccard treino final: {final_train_iou * 100:.2f}%")
    print(f"IoU/Jaccard validação final: {final_val_iou * 100:.2f}%")

    print("\n===== MELHORES RESULTADOS =====")
    print(
        f"Melhor accuracy validação: "
        f"{best_val_accuracy * 100:.2f}% "
        f"na época {best_val_accuracy_epoch}"
    )
    print(
        f"Melhor IoU/Jaccard validação: "
        f"{best_val_iou * 100:.2f}% "
        f"na época {best_val_iou_epoch}"
    )

    with open("resultados_treino.txt", "w", encoding="utf-8") as file:
        file.write("===== RESULTADOS FINAIS =====\n")
        file.write(
            f"Accuracy treino final: "
            f"{final_train_accuracy * 100:.2f}%\n"
        )
        file.write(
            f"Accuracy validação final: "
            f"{final_val_accuracy * 100:.2f}%\n"
        )
        file.write(
            f"IoU/Jaccard treino final: "
            f"{final_train_iou * 100:.2f}%\n"
        )
        file.write(
            f"IoU/Jaccard validação final: "
            f"{final_val_iou * 100:.2f}%\n\n"
        )

        file.write("===== MELHORES RESULTADOS =====\n")
        file.write(
            f"Melhor accuracy validação: "
            f"{best_val_accuracy * 100:.2f}% "
            f"na época {best_val_accuracy_epoch}\n"
        )
        file.write(
            f"Melhor IoU/Jaccard validação: "
            f"{best_val_iou * 100:.2f}% "
            f"na época {best_val_iou_epoch}\n"
        )

    print("\nResultados guardados em: resultados_treino.txt")


def evaluate_model_with_mean_iou(
    model,
    test_images,
    test_masks,
):
    predictions = model.predict(test_images)
    predicted_masks = np.argmax(predictions, axis=-1)

    print("\nClasses reais no teste:", np.unique(test_masks))
    print("Classes previstas pelo modelo:", np.unique(predicted_masks))

    mean_iou = MeanIoU(num_classes=NUMBER_OF_CLASSES)
    mean_iou.update_state(test_masks, predicted_masks)

    print(
        "\nMean IoU no conjunto de teste:",
        mean_iou.result().numpy(),
    )

    return predicted_masks


def show_prediction_examples(
    test_images,
    test_masks,
    predicted_masks,
    number_of_examples=4,
):
    number_of_examples = min(number_of_examples, len(test_images))

    selected_indexes = random.sample(
        range(len(test_images)),
        number_of_examples,
    )

    for index in selected_indexes:
        plt.figure(figsize=(16, 5))

        plt.subplot(1, 3, 1)
        plt.title(f"Imagem original - teste {index}")
        plt.imshow(test_images[index])
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Máscara correta/anotada")
        plt.imshow(
            test_masks[index],
            cmap=SEGMENTATION_CMAP,
            vmin=0,
            vmax=NUMBER_OF_CLASSES - 1,
        )
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Previsão do modelo")
        plt.imshow(
            predicted_masks[index],
            cmap=SEGMENTATION_CMAP,
            vmin=0,
            vmax=NUMBER_OF_CLASSES - 1,
        )
        plt.axis("off")
        add_class_legend()

        plt.tight_layout()
        plt.show()


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    print("Pasta de imagens:", IMAGES_DIR)
    print("Pasta de máscaras:", MASKS_DIR)

    pairs = find_image_mask_pairs()

    train_pairs, validation_pairs, test_pairs = split_image_mask_pairs(
        pairs,
    )

    train_images, train_masks = prepare_dataset_from_pairs(
        train_pairs,
        "treino",
    )

    validation_images, validation_masks = prepare_dataset_from_pairs(
        validation_pairs,
        "validação",
    )

    test_images, test_masks = prepare_dataset_from_pairs(
        test_pairs,
        "teste",
    )

    print("\nResumo final:")

    print(
        f"Treino: {train_images.shape}, {train_masks.shape}"
    )

    print(
        f"Validação: {validation_images.shape}, "
        f"{validation_masks.shape}"
    )

    print(
        f"Teste: {test_images.shape}, {test_masks.shape}"
    )

    print("\nClasses no treino:", np.unique(train_masks))
    print("Classes na validação:", np.unique(validation_masks))
    print("Classes no teste:", np.unique(test_masks))

    show_random_examples(
        train_images,
        train_masks,
        number_of_examples=15,
    )

    # ========================================================
    # TREINO DA U-NET
    # ========================================================

    model, history = train_unet_model(
        train_images=train_images,
        train_masks=train_masks,
        validation_images=validation_images,
        validation_masks=validation_masks,
        epochs=60,
        batch_size=4,
    )

    plot_training_history(history)

    save_training_results(history)

    model.save(MODELS_DIR / "ultimo_modelo_unet.keras")

    print("\nÚltimo modelo guardado como: modelos/ultimo_modelo_unet.keras")
    print("Melhor modelo guardado como: modelos/melhor_modelo_unet.keras")

    # ========================================================
    # AVALIAÇÃO COM O MELHOR MODELO
    # ========================================================

    best_model = load_model(
        MODELS_DIR / "melhor_modelo_unet.keras",
        custom_objects={"jacard_coef": jacard_coef},
    )

    predicted_test_masks = evaluate_model_with_mean_iou(
        model=best_model,
        test_images=test_images,
        test_masks=test_masks,
    )

    show_prediction_examples(
        test_images=test_images,
        test_masks=test_masks,
        predicted_masks=predicted_test_masks,
        number_of_examples=4,
    )