import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import random
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path("treino")

IMAGES_DIR = BASE_DIR / "imagens"
MASKS_DIR = BASE_DIR / "mascaras"
print(IMAGES_DIR)

PATCH_SIZE = 256

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ============================================================
# CLASSES DA SEGMENTAÇÃO
# ============================================================
# Substitui estas cores pelas cores RGB usadas nas tuas máscaras.
#
# Exemplo:
# "estrada": (255, 0, 0)
#
# O primeiro número é vermelho, o segundo verde e o terceiro azul.

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


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def list_files(folder: Path, valid_extensions: set[str]) -> list[Path]:
    """
    Devolve os ficheiros existentes numa pasta com extensões válidas.
    """

    if not folder.exists():
        raise FileNotFoundError(f"A pasta não existe: {folder}")

    files = [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in valid_extensions
    ]

    return sorted(files)


def create_file_map(files: list[Path]) -> dict[str, Path]:
    """
    Cria um dicionário em que a chave é o nome do ficheiro sem extensão.

    Exemplo:
        imagem_01.jpg -> {"imagem_01": caminho}
    """

    return {file.stem: file for file in files}


def find_image_mask_pairs() -> list[tuple[Path, Path]]:
    """
    Associa imagens e máscaras pelo nome-base do ficheiro.
    """

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
    """
    Lê uma imagem com OpenCV e converte-a de BGR para RGB.
    """

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_rgb_mask(mask_path: Path) -> np.ndarray:
    """
    Lê uma máscara colorida e converte-a de BGR para RGB.
    """

    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)

    if mask_bgr is None:
        raise ValueError(f"Não foi possível abrir a máscara: {mask_path}")

    return cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)


def crop_to_patch_size(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorta a imagem e a máscara para dimensões divisíveis pelo patch_size.

    O recorte é feito a partir do canto superior esquerdo.
    """

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
            f"A imagem tem dimensões inferiores a {patch_size} × {patch_size}: "
            f"{width} × {height}"
        )

    cropped_image = image[:cropped_height, :cropped_width]
    cropped_mask = mask[:cropped_height, :cropped_width]

    return cropped_image, cropped_mask


def rgb_mask_to_class_mask(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Converte uma máscara RGB numa máscara de classes.

    Resultado:
        background      -> 0
        campo_agricola  -> 1
        floresta        -> 2
        agua            -> 3
        estrada         -> 4
        edificio        -> 5
    """

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
            f"Aviso: foram encontrados {unrecognized_count} píxeis "
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
    """
    Divide uma imagem e a respetiva máscara em patches sem sobreposição.
    """

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
            # O patchify acrescenta uma dimensão extra às imagens RGB.
            image_patch = image_patches[row, column, 0]
            mask_patch = mask_patches[row, column]

            # Normalização para valores entre 0 e 1.
            image_patch = image_patch.astype(np.float32) / 255.0

            # A máscara mantém valores inteiros das classes.
            mask_patch = mask_patch.astype(np.uint8)

            image_patch_list.append(image_patch)
            mask_patch_list.append(mask_patch)

    return image_patch_list, mask_patch_list


def show_example(
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
) -> None:
    """
    Mostra um patch da imagem e a respetiva máscara.
    """

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Imagem")
    plt.imshow(image_patch)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Máscara por classes")
    plt.imshow(
        mask_patch,
        cmap="tab10",
        vmin=0,
        vmax=NUMBER_OF_CLASSES - 1,
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# CARREGAMENTO DO DATASET
# ============================================================

def prepare_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega todos os pares imagem/máscara e cria os patches.
    """

    pairs = find_image_mask_pairs()

    image_dataset = []
    mask_dataset = []

    print(f"\nForam encontrados {len(pairs)} pares imagem/máscara.\n")

    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        print(
            f"[{index}/{len(pairs)}] "
            f"Imagem: {image_path.name} | Máscara: {mask_path.name}"
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

        print(f"  Patches criados: {len(image_patches)}")

    images_array = np.asarray(image_dataset, dtype=np.float32)
    masks_array = np.asarray(mask_dataset, dtype=np.uint8)

    print("\nDataset preparado com sucesso.")
    print(f"Formato das imagens: {images_array.shape}")
    print(f"Formato das máscaras: {masks_array.shape}")
    print(f"Número de classes: {NUMBER_OF_CLASSES}")
    print(f"Classes encontradas: {np.unique(masks_array)}")

    return images_array, masks_array


def show_random_examples(images, masks, number_of_examples=4):
    number_of_examples = min(number_of_examples, len(images))

    selected_indexes = random.sample(
        range(len(images)),
        number_of_examples,
    )

    for index in selected_indexes:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title(f"Imagem - patch {index}")
        plt.imshow(images[index])
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Máscara - patch {index}")
        plt.imshow(
            masks[index],
            cmap="tab10",
            vmin=0,
            vmax=NUMBER_OF_CLASSES - 1,
        )
        plt.axis("off")

        plt.tight_layout()
        plt.show()




def split_image_mask_pairs(
    pairs,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
):
    """
    Divide os pares imagem/máscara antes da criação dos patches.
    """

    if not np.isclose(
        train_ratio + validation_ratio + test_ratio,
        1.0,
    ):
        raise ValueError(
            "As percentagens de treino, validação e teste "
            "devem somar 1."
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
    """
    Carrega os pares recebidos e cria patches.
    """

    image_dataset = []
    mask_dataset = []

    print(f"\nA preparar o conjunto de {dataset_name}...")

    for index, (image_path, mask_path) in enumerate(
        pairs,
        start=1,
    ):
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

    images_array = np.asarray(
        image_dataset,
        dtype=np.float32,
    )

    masks_array = np.asarray(
        mask_dataset,
        dtype=np.uint8,
    )

    print(
        f"{dataset_name}: "
        f"{len(images_array)} patches criados."
    )

    print(
        f"Formato das imagens: {images_array.shape}"
    )

    print(
        f"Formato das máscaras: {masks_array.shape}"
    )

    return images_array, masks_array

# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    pairs = find_image_mask_pairs()

    train_pairs, validation_pairs, test_pairs = (
        split_image_mask_pairs(pairs)
    )

    train_images, train_masks = prepare_dataset_from_pairs(
        train_pairs,
        "treino",
    )

    validation_images, validation_masks = (
        prepare_dataset_from_pairs(
            validation_pairs,
            "validação",
        )
    )

    test_images, test_masks = prepare_dataset_from_pairs(
        test_pairs,
        "teste",
    )

    print("\nResumo final:")

    print(
        f"Treino: "
        f"{train_images.shape}, {train_masks.shape}"
    )

    print(
        f"Validação: "
        f"{validation_images.shape}, "
        f"{validation_masks.shape}"
    )

    print(
        f"Teste: "
        f"{test_images.shape}, {test_masks.shape}"
    )

    print(
        "\nClasses no treino:",
        np.unique(train_masks),
    )

    print(
        "Classes na validação:",
        np.unique(validation_masks),
    )

    print(
        "Classes no teste:",
        np.unique(test_masks),
    )

    show_random_examples(
        train_images,
        train_masks,
        number_of_examples=4,
    )
        