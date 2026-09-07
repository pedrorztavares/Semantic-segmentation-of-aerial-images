import random # escolhe exemplos aleatórios para mostrar
from pathlib import Path #

import cv2 #abre as imagens e converte de BGR para RGB
import numpy as np
import matplotlib.pyplot as plt # serve para mostrar as imagens e máscaras
from matplotlib.colors import ListedColormap #serve para criar uma colormap personalizada para mostrar as máscaras
from matplotlib.patches import Patch # serve para criar a legenda das classes na máscara

from patchify import patchify # divide as imagens grandes em blocos mais pequenos (patches) para treinar a rede neural
from sklearn.model_selection import train_test_split # divide o dataset em treino, validação e teste

import tensorflow as tf # ferramentas principais para treinar a rede neural
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

BASE_DIR = Path("treino") #pasta base 

IMAGES_DIR = BASE_DIR / "imagens" #pasta onde estão as imagens de treino
MASKS_DIR = BASE_DIR / "mascaras"#pasta onde estão as máscaras de treino

MODELS_DIR = Path("modelos") #pasta dos odelos treinados
MODELS_DIR.mkdir(exist_ok=True) # cria a pasta "modelos" se não existir

PATCH_SIZE = 256 

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"} #extensões aceites para as imagens
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ============================================================
# CLASSES DA SEGMENTAÇÃO
# ============================================================

CLASS_COLORS = { #cada classe tem uma cor RGB associada, que é usada nas máscaras
    "background": (0, 0, 0), #preto
    "campo_agricola": (210, 210, 24),#amarelo
    "floresta": (36, 179, 83),#verde
    "agua": (45, 12, 212),#azul escuro
    "estrada": (255, 255, 255),#branco
    "edificio": (245, 147, 49),#laranja
}

CLASS_NAMES = list(CLASS_COLORS.keys()) #vai buscar os nomes das classes e guarda numa lista
NUMBER_OF_CLASSES = len(CLASS_NAMES) # vê o comprimento da lista de classes, que é o número de classes

# Classe a ignorar nas métricas e na loss
IGNORE_CLASS_ID = 0 #ignora o background nas métricas e na loss, para não afetar a performance do modelo

SEGMENTATION_CMAP = ListedColormap( #cria um mapa de cores para mostrar as máscaras, com as cores das classes 
    [
        tuple(channel / 255.0 for channel in CLASS_COLORS[name])
        for name in CLASS_NAMES
    ]
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def list_files(folder: Path, valid_extensions: set[str]) -> list[Path]: # procura ficheiros numa pasta e devolve uma lista com os ficheiros encontrados
    if not folder.exists():
        raise FileNotFoundError(f"A pasta não existe: {folder}")

    files = [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in valid_extensions #seleciona apenas os ficheiros com as extensões válidas
    ]

    return sorted(files) #retorna a lista de ficheiros ordenada por ordem alfabética


def create_file_map(files: list[Path]) -> dict[str, Path]:
    return {file.stem: file for file in files} #retorna um dicionário apenas com os nomes base do ficheiro sem extensão .png...


def find_image_mask_pairs() -> list[tuple[Path, Path]]: # procura a imagem a respetiva máscara e devolve uma lista de pares (imagem, máscara)
    image_files = list_files(IMAGES_DIR, IMAGE_EXTENSIONS)
    mask_files = list_files(MASKS_DIR, MASK_EXTENSIONS)

    image_map = create_file_map(image_files)
    mask_map = create_file_map(mask_files)

    common_names = sorted(set(image_map) & set(mask_map)) #compara os nomes para encontrar os pares correspondentes de imagem e máscara

    images_without_mask = sorted(set(image_map) - set(mask_map)) #exclusão de imagens que não têm máscara correspondente
    masks_without_image = sorted(set(mask_map) - set(image_map))#exclusão de máscaras que não têm imagem correspondente

    if images_without_mask:
        print("\nImagens sem máscara correspondente:") #identifica imagens sem mascara
        for name in images_without_mask:
            print(f"  - {image_map[name].name}")

    if masks_without_image:
        print("\nMáscaras sem imagem correspondente:") #identifcia máscaras sem imagem
        for name in masks_without_image:
            print(f"  - {mask_map[name].name}")

    if not common_names:
        raise RuntimeError(
            "Não foi encontrado nenhum par imagem/máscara.\n" #interrompe o programa se não houverem pares
            "Confirma se os ficheiros têm o mesmo nome-base."
        )

    return [(image_map[name], mask_map[name]) for name in common_names] #retorna a lista de pares, se existirem, com o nome da imagem e o nome da máscara correspondente


def load_rgb_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR) #lê a imagem usando o formato BGR do OpenCV

    if image_bgr is None:
        raise ValueError(f"Não foi possível abrir a imagem: {image_path}") #interrompe o programa se a imagem não puder ser aberta

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) #converte a imagem de BGR para RGB


def load_rgb_mask(mask_path: Path) -> np.ndarray:
    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR) #lê a máscara utilizando o formato BGR do OpenCV

    if mask_bgr is None:
        raise ValueError(f"Não foi possível abrir a máscara: {mask_path}") #interrompe o programa se a máscara não puder ser aberta

    return cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB) #converte a máscara de BGR para RGB


def crop_to_patch_size(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if image.shape[:2] != mask.shape[:2]: #verifica se a imagem e a máscara têm a mesma altura e largura
        raise ValueError(
            "A imagem e a máscara têm dimensões diferentes: "
            f"imagem={image.shape[:2]}, máscara={mask.shape[:2]}"
        )

    height, width = image.shape[:2] #obtém a altura e a largura da imagem

    cropped_height = (height // patch_size) * patch_size #calcula a maior altura divisível pelo tamanho do patch
    cropped_width = (width // patch_size) * patch_size #calcula a maior largura divisível pelo tamanho do patch

    if cropped_height == 0 or cropped_width == 0: #verifica se a imagem é menor do que o tamanho do patch
        raise ValueError(
            f"A imagem tem dimensões inferiores a {patch_size}x{patch_size}: "
            f"{width}x{height}"
        )

    cropped_image = image[:cropped_height, :cropped_width] #corta a imagem para uma dimensão divisível pelo patch
    cropped_mask = mask[:cropped_height, :cropped_width] #corta a máscara da mesma forma que a imagem

    return cropped_image, cropped_mask #retorna a imagem e a máscara já cortadas


def rgb_mask_to_class_mask(mask_rgb: np.ndarray) -> np.ndarray:
    class_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8) #cria uma máscara numérica inicialmente preenchida com a classe background
    recognized_pixels = np.zeros(mask_rgb.shape[:2], dtype=bool) #guarda quais pixels têm uma cor reconhecida

    for class_id, class_name in enumerate(CLASS_NAMES): #percorre todas as classes e os respetivos identificadores
        class_color = np.array(CLASS_COLORS[class_name], dtype=np.uint8) #converte a cor da classe para um array NumPy

        pixels_of_class = np.all(mask_rgb == class_color, axis=-1) #identifica os pixels que têm exatamente a cor da classe

        class_mask[pixels_of_class] = class_id #atribui o número da classe aos pixels correspondentes
        recognized_pixels |= pixels_of_class #marca estes pixels como reconhecidos

    unrecognized_count = np.count_nonzero(~recognized_pixels) #conta os pixels que não correspondem a nenhuma cor definida

    if unrecognized_count > 0: #mostra um aviso se existirem cores não definidas
        unique_unknown_colors = np.unique( #obtém as cores desconhecidas sem repetir valores
            mask_rgb[~recognized_pixels].reshape(-1, 3),
            axis=0,
        )

        print(
            f"\nAviso: foram encontrados {unrecognized_count} píxeis "
            "com cores não definidas."
        )

        print("Algumas cores RGB não reconhecidas:")

        for color in unique_unknown_colors[:10]: #mostra no máximo as primeiras dez cores desconhecidas
            print(f"  - {tuple(int(value) for value in color)}") #apresenta a cor RGB encontrada

    return class_mask #retorna a máscara com números em vez de cores RGB


def create_patches(
    image: np.ndarray,
    class_mask: np.ndarray,
    patch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    image_patches = patchify( #divide a imagem em patches com três canais de cor RGB
        image,
        (patch_size, patch_size, 3),
        step=patch_size,
    )

    mask_patches = patchify( #divide a máscara em patches bidimensionais correspondentes
        class_mask,
        (patch_size, patch_size),
        step=patch_size,
    )

    image_patch_list = [] #cria a lista que irá guardar os patches das imagens
    mask_patch_list = [] #cria a lista que irá guardar os patches das máscaras

    number_of_rows = image_patches.shape[0] #obtém o número de linhas de patches
    number_of_columns = image_patches.shape[1] #obtém o número de colunas de patches

    for row in range(number_of_rows): #percorre as linhas de patches
        for column in range(number_of_columns): #percorre as colunas de patches
            image_patch = image_patches[row, column, 0] #obtém o patch da imagem na posição atual
            mask_patch = mask_patches[row, column] #obtém o patch correspondente da máscara

            image_patch = image_patch.astype(np.float32) / 255.0 #normaliza os pixels da imagem para o intervalo entre 0 e 1
            mask_patch = mask_patch.astype(np.uint8) #garante que os identificadores das classes são inteiros

            image_patch_list.append(image_patch) #adiciona o patch da imagem à lista
            mask_patch_list.append(mask_patch) #adiciona o patch da máscara à lista

    return image_patch_list, mask_patch_list #retorna as duas listas de patches


def split_image_mask_pairs(
    pairs,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
):
    if not np.isclose( #confirma que as percentagens definidas totalizam 100%
        train_ratio + validation_ratio + test_ratio,
        1.0,
    ):
        raise ValueError(
            "As percentagens de treino, validação e teste devem somar 1."
        )

    train_pairs, temporary_pairs = train_test_split( #separa os dados de treino dos dados restantes
        pairs,
        test_size=validation_ratio + test_ratio,
        random_state=random_state,
        shuffle=True,
    )

    test_fraction_of_temporary = ( #calcula que parte dos dados restantes pertence ao teste
        test_ratio / (validation_ratio + test_ratio)
    )

    validation_pairs, test_pairs = train_test_split( #divide os dados restantes entre validação e teste
        temporary_pairs,
        test_size=test_fraction_of_temporary,
        random_state=random_state,
        shuffle=True,
    )

    print("\nDivisão do dataset:")
    print(f"Treino: {len(train_pairs)} imagens")
    print(f"Validação: {len(validation_pairs)} imagens")
    print(f"Teste: {len(test_pairs)} imagens")

    return train_pairs, validation_pairs, test_pairs #retorna os três conjuntos de dados


def prepare_dataset_from_pairs(pairs, dataset_name):
    image_dataset = [] #lista que irá guardar todos os patches das imagens
    mask_dataset = [] #lista que irá guardar todos os patches das máscaras

    print(f"\nA preparar o conjunto de {dataset_name}...")

    for index, (image_path, mask_path) in enumerate(pairs, start=1): #processa cada par de imagem e máscara
        print(
            f"[{index}/{len(pairs)}] "
            f"{image_path.name} | {mask_path.name}"
        )

        image = load_rgb_image(image_path) #carrega a imagem no formato RGB
        mask_rgb = load_rgb_mask(mask_path) #carrega a máscara no formato RGB

        image, mask_rgb = crop_to_patch_size( #ajusta as dimensões da imagem e da máscara
            image,
            mask_rgb,
            PATCH_SIZE,
        )

        class_mask = rgb_mask_to_class_mask(mask_rgb) #converte as cores da máscara em identificadores numéricos

        image_patches, mask_patches = create_patches( #divide a imagem e a máscara em patches
            image,
            class_mask,
            PATCH_SIZE,
        )

        image_dataset.extend(image_patches) #adiciona os patches da imagem ao dataset
        mask_dataset.extend(mask_patches) #adiciona os patches da máscara ao dataset

    images_array = np.asarray(image_dataset, dtype=np.float32) #converte os patches das imagens num array NumPy
    masks_array = np.asarray(mask_dataset, dtype=np.uint8) #converte os patches das máscaras num array NumPy

    print(f"{dataset_name}: {len(images_array)} patches criados.")
    print(f"Formato das imagens: {images_array.shape}")
    print(f"Formato das máscaras: {masks_array.shape}")
    print(f"Classes encontradas: {np.unique(masks_array)}")

    return images_array, masks_array #retorna as imagens e máscaras preparadas


def add_class_legend():
    legend_handles = [ #cria um item de legenda para cada classe
        Patch(
            facecolor=tuple(channel / 255.0 for channel in CLASS_COLORS[name]),
            edgecolor="black",
            label=name,
        )
        for name in CLASS_NAMES
    ]

    plt.legend( #adiciona a legenda ao gráfico
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize="small",
    )


def show_random_examples(images, masks, number_of_examples=5):
    number_of_examples = min(number_of_examples, len(images)) #evita pedir mais exemplos do que os existentes

    selected_indexes = random.sample( #seleciona índices aleatórios sem repetir exemplos
        range(len(images)),
        number_of_examples,
    )

    for index in selected_indexes: #mostra cada exemplo selecionado
        plt.figure(figsize=(11, 5)) #cria uma figura para a imagem e a máscara

        plt.subplot(1, 2, 1) #cria o espaço para mostrar a imagem
        plt.title(f"Imagem - patch {index}") #define o título da imagem
        plt.imshow(images[index]) #mostra o patch da imagem
        plt.axis("off") #esconde os eixos do gráfico

        plt.subplot(1, 2, 2) #cria o espaço para mostrar a máscara
        plt.title(f"Máscara - patch {index}") #define o título da máscara
        plt.imshow( #mostra a máscara com as cores das classes
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
# LOSS E MÉTRICAS SEM BACKGROUND
# ============================================================

def masked_categorical_crossentropy(y_true, y_pred):
    y_true_ids = tf.argmax(y_true, axis=-1) #converte a máscara one-hot no identificador de cada classe

    valid_mask = tf.not_equal(y_true_ids, IGNORE_CLASS_ID) #seleciona apenas os pixels que não são background
    valid_mask = tf.cast(valid_mask, tf.float32) #converte os valores booleanos para números

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred) #calcula o erro de classificação de cada pixel

    loss = loss * valid_mask #ignora o erro dos pixels pertencentes ao background

    return tf.reduce_sum(loss) / (tf.reduce_sum(valid_mask) + K.epsilon()) #calcula a média do erro apenas nos pixels válidos


def masked_accuracy(y_true, y_pred):
    y_true_ids = tf.argmax(y_true, axis=-1) #obtém as classes reais dos pixels
    y_pred_ids = tf.argmax(y_pred, axis=-1) #obtém as classes previstas pelo modelo

    valid_mask = tf.not_equal(y_true_ids, IGNORE_CLASS_ID) #seleciona os pixels que não são background

    correct_predictions = tf.equal(y_true_ids, y_pred_ids) #verifica quais previsões estão corretas
    correct_predictions = tf.logical_and(correct_predictions, valid_mask) #ignora as previsões sobre o background

    correct_predictions = tf.cast(correct_predictions, tf.float32) #converte os resultados para números
    valid_mask = tf.cast(valid_mask, tf.float32) #converte a máscara de pixels válidos para números

    return tf.reduce_sum(correct_predictions) / ( #calcula a percentagem de previsões corretas sem background
        tf.reduce_sum(valid_mask) + K.epsilon()
    )


def masked_jaccard_coef(y_true, y_pred):
    y_true_ids = tf.argmax(y_true, axis=-1) #obtém as classes reais dos pixels
    y_pred_ids = tf.argmax(y_pred, axis=-1) #obtém as classes previstas pelo modelo

    iou_values = [] #lista que irá guardar o IoU de cada classe

    for class_id in range(1, NUMBER_OF_CLASSES): #percorre as classes sem incluir o background
        true_class = tf.equal(y_true_ids, class_id) #identifica os pixels reais da classe atual
        pred_class = tf.equal(y_pred_ids, class_id) #identifica os pixels previstos como a classe atual

        intersection = tf.logical_and(true_class, pred_class) #pixels que pertencem à classe real e prevista
        union = tf.logical_or(true_class, pred_class) #pixels que pertencem à classe real ou prevista

        intersection = tf.reduce_sum(tf.cast(intersection, tf.float32)) #conta os pixels da interseção
        union = tf.reduce_sum(tf.cast(union, tf.float32)) #conta os pixels da união

        iou = intersection / (union + K.epsilon()) #calcula o IoU da classe atual

        iou_values.append(iou) #guarda o resultado da classe atual

    return tf.reduce_mean(tf.stack(iou_values)) #retorna o IoU médio entre todas as classes


# ============================================================
# MODELO U-NET
# ============================================================

def build_unet_model(
    image_height: int,
    image_width: int,
    image_channels: int,
    number_of_classes: int,
) -> Model:
    inputs = Input((image_height, image_width, image_channels)) #define a entrada da rede com o tamanho dos patches

    # Encoder: reduz a resolução e aprende características da imagem.
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

    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(p4) #camada mais profunda da rede, com características mais abstratas
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(c5)

    # Decoder: aumenta a resolução e recupera os detalhes espaciais da imagem.
    u6 = Conv2DTranspose( #aumenta a resolução das características aprendidas
        128,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(c5)

    u6 = concatenate([u6, c4]) #liga o decoder às características correspondentes do encoder
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(c6)

    u7 = Conv2DTranspose( #continua a aumentar a resolução da imagem
        64,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(c6)

    u7 = concatenate([u7, c3]) #combina características do decoder e do encoder
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(c7)

    u8 = Conv2DTranspose( #aumenta novamente a resolução das características
        32,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(c7)

    u8 = concatenate([u8, c2]) #recupera detalhes da segunda camada do encoder
    c8 = Conv2D(32, (3, 3), activation="relu", padding="same")(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation="relu", padding="same")(c8)

    u9 = Conv2DTranspose( #recupera a resolução original do patch
        16,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(c8)

    u9 = concatenate([u9, c1]) #liga as características finais às primeiras características extraídas
    c9 = Conv2D(16, (3, 3), activation="relu", padding="same")(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation="relu", padding="same")(c9)

    outputs = Conv2D( #gera uma probabilidade para cada classe em cada pixel
        number_of_classes,
        (1, 1),
        activation="softmax",
    )(c9)

    model = Model(inputs=[inputs], outputs=[outputs]) #cria o modelo completo entre a entrada e a saída

    return model #retorna a U-Net construída


def train_unet_model(
    train_images,
    train_masks,
    validation_images,
    validation_masks,
    epochs=120,
    batch_size=4,
):
    train_masks_cat = to_categorical( #converte as máscaras de treino para o formato one-hot
        train_masks,
        num_classes=NUMBER_OF_CLASSES,
    )

    validation_masks_cat = to_categorical( #converte as máscaras de validação para o formato one-hot
        validation_masks,
        num_classes=NUMBER_OF_CLASSES,
    )

    image_height = train_images.shape[1] #obtém a altura dos patches
    image_width = train_images.shape[2] #obtém a largura dos patches
    image_channels = train_images.shape[3] #obtém o número de canais de cor

    model = build_unet_model( #cria uma nova U-Net com as dimensões dos dados
        image_height=image_height,
        image_width=image_width,
        image_channels=image_channels,
        number_of_classes=NUMBER_OF_CLASSES,
    )

    model.compile( #configura o otimizador, a função de erro e as métricas
        optimizer=Adam(learning_rate=0.001),
        loss=masked_categorical_crossentropy,
        metrics=[masked_accuracy, masked_jaccard_coef],
    )

    model.summary() #mostra a estrutura e o número de parâmetros do modelo

    checkpoint = ModelCheckpoint( #cria um callback para guardar o melhor modelo
        filepath=MODELS_DIR / "melhor_modelo_unet.keras",
        monitor="val_masked_jaccard_coef",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.fit( #inicia o treino do modelo com os dados de treino e validação
        train_images,
        train_masks_cat,
        validation_data=(validation_images, validation_masks_cat),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint],
    )

    return model, history #retorna o modelo treinado e o histórico dos resultados


# ============================================================
# GRÁFICOS E AVALIAÇÃO
# ============================================================

def plot_training_history(history):
    plt.figure(figsize=(8, 5)) #cria a figura do gráfico da loss
    plt.plot(history.history["loss"], label="Loss treino") #mostra a loss do treino em cada época
    plt.plot(history.history["val_loss"], label="Loss validação") #mostra a loss da validação em cada época
    plt.title("Training and validation loss")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.show() #apresenta o gráfico

    plt.figure(figsize=(8, 5)) #cria a figura do gráfico da accuracy
    plt.plot(history.history["masked_accuracy"], label="Accuracy treino sem background") #mostra a accuracy do treino
    plt.plot( #mostra a accuracy da validação
        history.history["val_masked_accuracy"],
        label="Accuracy validação sem background",
    )
    plt.title("Training and validation accuracy sem background")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show() #apresenta o gráfico

    plt.figure(figsize=(8, 5)) #cria a figura do gráfico do IoU
    plt.plot( #mostra o IoU do treino
        history.history["masked_jaccard_coef"],
        label="IoU/Jaccard treino sem background",
    )
    plt.plot( #mostra o IoU da validação
        history.history["val_masked_jaccard_coef"],
        label="IoU/Jaccard validação sem background",
    )
    plt.title("Training and validation IoU/Jaccard sem background")
    plt.xlabel("Épocas")
    plt.ylabel("IoU/Jaccard")
    plt.legend()
    plt.show() #apresenta o gráfico


def save_training_results(history):
    final_train_accuracy = history.history["masked_accuracy"][-1] #obtém a accuracy do treino na última época
    final_val_accuracy = history.history["val_masked_accuracy"][-1] #obtém a accuracy da validação na última época
    final_train_iou = history.history["masked_jaccard_coef"][-1] #obtém o IoU do treino na última época
    final_val_iou = history.history["val_masked_jaccard_coef"][-1] #obtém o IoU da validação na última época

    best_val_accuracy = max(history.history["val_masked_accuracy"]) #encontra a melhor accuracy da validação
    best_val_iou = max(history.history["val_masked_jaccard_coef"]) #encontra o melhor IoU da validação

    best_val_accuracy_epoch = (
        history.history["val_masked_accuracy"].index(best_val_accuracy) + 1
    )

    best_val_iou_epoch = (
        history.history["val_masked_jaccard_coef"].index(best_val_iou) + 1
    )

    print("\n===== RESULTADOS FINAIS SEM BACKGROUND =====")
    print(f"Accuracy treino final: {final_train_accuracy * 100:.2f}%")
    print(f"Accuracy validação final: {final_val_accuracy * 100:.2f}%")
    print(f"IoU/Jaccard treino final: {final_train_iou * 100:.2f}%")
    print(f"IoU/Jaccard validação final: {final_val_iou * 100:.2f}%")

    print("\n===== MELHORES RESULTADOS SEM BACKGROUND =====")
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

    with open("resultados_treino.txt", "w", encoding="utf-8") as file: #cria ou substitui o ficheiro com os resultados
        file.write("===== RESULTADOS FINAIS SEM BACKGROUND =====\n") #guarda os resultados finais
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

        file.write("===== MELHORES RESULTADOS SEM BACKGROUND =====\n")
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
    predictions = model.predict(test_images) #gera as probabilidades previstas para cada classe e pixel
    predicted_masks = np.argmax(predictions, axis=-1) #escolhe a classe com maior probabilidade em cada pixel

    print("\nClasses reais no teste:", np.unique(test_masks))
    print("Classes previstas pelo modelo:", np.unique(predicted_masks))

    # Mean IoU normal, incluindo background
    mean_iou_all = MeanIoU(num_classes=NUMBER_OF_CLASSES) #cria a métrica IoU incluindo o background
    mean_iou_all.update_state(test_masks, predicted_masks) #compara as máscaras reais com as previstas

    print(
        "\nMean IoU no conjunto de teste incluindo background:",
        mean_iou_all.result().numpy(),
    )

    # Mean IoU manual sem background
    iou_values = [] #lista que irá guardar o IoU de cada classe sem background

    for class_id in range(1, NUMBER_OF_CLASSES): #calcula o IoU individual de cada classe sem background
        true_class = test_masks == class_id #seleciona os pixels reais da classe atual
        pred_class = predicted_masks == class_id #seleciona os pixels previstos como a classe atual

        intersection = np.logical_and(true_class, pred_class).sum() #conta os pixels corretamente classificados
        union = np.logical_or(true_class, pred_class).sum() #conta todos os pixels pertencentes à classe real ou prevista

        if union > 0: #só calcula o IoU se a classe aparecer na realidade ou na previsão
            iou = intersection / union #calcula o IoU da classe atual
            iou_values.append(iou) #guarda o IoU para calcular a média depois

            print(
                f"IoU classe {class_id} ({CLASS_NAMES[class_id]}): "
                f"{iou * 100:.2f}%"
            )

    if iou_values: #verifica se pelo menos uma classe foi encontrada
        mean_iou_no_background = np.mean(iou_values) #calcula a média dos IoUs das classes encontradas
        print(
            "\nMean IoU no conjunto de teste sem background:",
            mean_iou_no_background,
        )

    return predicted_masks #retorna as máscaras previstas para serem apresentadas


def show_prediction_examples(
    test_images,
    test_masks,
    predicted_masks,
    number_of_examples=4,
):
    number_of_examples = min(number_of_examples, len(test_images)) #limita o número de exemplos aos testes existentes

    selected_indexes = random.sample( #seleciona exemplos de teste aleatoriamente
        range(len(test_images)),
        number_of_examples,
    )

    for index in selected_indexes: #mostra cada previsão selecionada
        plt.figure(figsize=(16, 5)) #cria uma figura com três imagens lado a lado

        plt.subplot(1, 3, 1) #cria o espaço para a imagem original
        plt.title(f"Imagem original - teste {index}") #define o título da imagem original
        plt.imshow(test_images[index]) #mostra a imagem original
        plt.axis("off") #esconde os eixos

        plt.subplot(1, 3, 2) #cria o espaço para a máscara correta
        plt.title("Máscara correta/anotada") #define o título da máscara correta
        plt.imshow( #mostra a máscara anotada com as cores das classes
            test_masks[index],
            cmap=SEGMENTATION_CMAP,
            vmin=0,
            vmax=NUMBER_OF_CLASSES - 1,
        )
        plt.axis("off")

        plt.subplot(1, 3, 3) #cria o espaço para a previsão do modelo
        plt.title("Previsão do modelo") #define o título da previsão
        plt.imshow( #mostra a máscara prevista pelo modelo
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
    print("Pasta de imagens:", IMAGES_DIR) #mostra o caminho da pasta das imagens
    print("Pasta de máscaras:", MASKS_DIR) #mostra o caminho da pasta das máscaras

    pairs = find_image_mask_pairs() #encontra os pares correspondentes de imagens e máscaras

    train_pairs, validation_pairs, test_pairs = split_image_mask_pairs( #divide os pares em treino, validação e teste
        pairs,
    )

    train_images, train_masks = prepare_dataset_from_pairs( #prepara os patches usados para treinar
        train_pairs,
        "treino",
    )

    validation_images, validation_masks = prepare_dataset_from_pairs( #prepara os patches usados para validar
        validation_pairs,
        "validação",
    )

    test_images, test_masks = prepare_dataset_from_pairs( #prepara os patches usados para testar
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

    show_random_examples( #mostra alguns exemplos do dataset antes do treino
        train_images,
        train_masks,
        number_of_examples=5,
    )

    # ========================================================
    # TREINO DA U-NET
    # ========================================================

    model, history = train_unet_model( #treina a U-Net e guarda o histórico do treino
        train_images=train_images,
        train_masks=train_masks,
        validation_images=validation_images,
        validation_masks=validation_masks,
        epochs=120,
        batch_size=4,
    )

    plot_training_history(history) #mostra a evolução das métricas durante o treino

    save_training_results(history) #guarda os resultados do treino num ficheiro de texto

    model.save(MODELS_DIR / "ultimo_modelo_unet.keras") #guarda o modelo obtido na última época

    print("\nÚltimo modelo guardado como: modelos/ultimo_modelo_unet.keras")
    print("Melhor modelo guardado como: modelos/melhor_modelo_unet.keras")

    # ========================================================
    # AVALIAÇÃO COM O MELHOR MODELO
    # ========================================================

    best_model = load_model( #carrega o modelo que obteve o melhor IoU na validação
        MODELS_DIR / "melhor_modelo_unet.keras",
        custom_objects={
            "masked_categorical_crossentropy": masked_categorical_crossentropy,
            "masked_accuracy": masked_accuracy,
            "masked_jaccard_coef": masked_jaccard_coef,
        },
    )

    predicted_test_masks = evaluate_model_with_mean_iou( #avalia o melhor modelo com os dados de teste
        model=best_model,
        test_images=test_images,
        test_masks=test_masks,
    )

    show_prediction_examples( #mostra as imagens, máscaras reais e previsões finais
        test_images=test_images,
        test_masks=test_masks,
        predicted_masks=predicted_test_masks,
        number_of_examples=20,
    )