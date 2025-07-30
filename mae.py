import numpy as np
import pandas as pd
import sys
import os
import subprocess
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
from skimage.transform import resize
import urllib.request
import torch
import timm
from timm.models.vision_transformer import PatchEmbed, Block
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from statistics import mean

def serie_to_2Dmatrix(serie):
    acf_vals = acf(serie, nlags=min(50, len(serie)//2), fft=True)

    lags = np.arange(1, len(acf_vals))
    acf_sem_lag0 = acf_vals[1:]
    peaks, _ = find_peaks(acf_sem_lag0, height=0.2)

    if len(peaks) == 0:
        P = 1
    else:
        P = peaks[0] + 1
    L = (len(serie) // P) * P
    serie_cortada = serie[:L]
    matriz_2d = serie_cortada.reshape(P, L // P)
    return matriz_2d


def get_Inorm(matriz_2d):
    desvio_padrao = np.std(matriz_2d)
    meann = np.mean(matriz_2d)
    r = 0.4
    Inorm = r * (matriz_2d - meann) / desvio_padrao
    return Inorm

def get_Igrey(Inorm, L, H):
    c = 0.4
    S = 1
    N = max(Inorm.shape)

    # Etapa 1: Render Igray (imagem 3 canais iguais)
    Igray = np.stack([Inorm] * 3, axis=-1)
    # Etapa 2: Alignment
    n = int(np.floor(c * N * L / (L + H)))  # número de patches visíveis na horizontal
    target_shape = (N * S, n * S)           # shape final da imagem após resize

    # Resize Igray (mantendo apenas 1 canal para simplificar o MAE)
    Igray_resized = resize(Inorm, target_shape, order=1, mode='reflect', anti_aliasing=False)

    # Converter para imagem 3 canais (opcional, dependendo do input do MAE)
    Igray_resized_3ch = np.stack([Igray_resized] * 3, axis=-1)
    return Igray_resized_3ch


def run_one_image_from_aligned_image(Igray_resized_3ch, model, L=300, H=100, c=0.4):
    """
    Igray_resized_3ch: imagem com 3 canais (já alinhada), shape (H, W, 3), valores já normalizados
    model: modelo MAE do repo da Meta (Vit-base etc)
    L, H, c: hiperparâmetros do VisionTS
    """
    # 1. Redimensionar para 224x224 (como esperado pelo MAE)
    img = resize(Igray_resized_3ch, (224, 224), order=1, mode='reflect', anti_aliasing=False)
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0)       # [1, H, W, C]
    if False:
        x = torch.einsum('nhwc->nchw', x).cuda()                      # [1, 3, 224, 224]
    if True:
        x = torch.einsum('nhwc->nchw', x).to(device)

    # 2. Criar máscara estruturada: lado direito mascarado
    patch_size = model.patch_embed.patch_size[0]         # geralmente 16
    num_patches_row = x.shape[2] // patch_size           # 224 / 16 = 14
    num_patches = num_patches_row ** 2                   # 14 x 14 = 196

    N = num_patches_row
    n = int(np.floor(c * N * L / (L + H)))                # número de colunas visíveis

    # Criar máscara: 0 = visível (esquerda), 1 = mascarado (direita)
    mask = torch.ones(num_patches, device=x.device)
    for row in range(N):
        for col in range(n):  # colunas visíveis
            idx = row * N + col
            mask[idx] = 0
    mask = mask.unsqueeze(0)  # [1, num_patches]

    # 3. Forward com máscara estruturada
    with torch.no_grad():
        if False:
            loss, y, _ = model(x.float(), mask=mask)         # usamos mask diretamente
        if True:
            loss, y, _ = model(x.float())
        y = model.unpatchify(y)                          # [1, 3, H, W]
        if False:
            y = torch.einsum('nchw->nhwc', y).cpu().numpy()
        if True:
            y = torch.einsum('nchw->nhwc', y).cpu()

        mask_recon = mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        mask_recon = model.unpatchify(mask_recon)
        mask_recon = torch.einsum('nchw->nhwc', mask_recon).cpu()

    x_vis = x.cpu()
    x_vis = torch.einsum('nchw->nhwc', x_vis)

    im_masked = x_vis * (1 - mask_recon)
    im_paste = x_vis * (1 - mask_recon) + y * mask_recon

    return y[0]  # retorno da imagem reconstruída (H, W, 3)


def reconstruct_and_evaluate_forecast(y_reconstructed, original_Inorm, L, H):
    """
    y_reconstructed: imagem reconstruída pelo MAE, shape (H, W, 3)
    original_Inorm: matriz 2D usada originalmente, shape (P, L/P)
    r: fator de normalização aplicado
    """
    r = 0.4
    import numpy as np
    from skimage.transform import resize
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # Etapa 1: média dos 3 canais para obter imagem grayscale reconstruída
    recon_gray = y_reconstructed.mean(axis=-1)  # shape (H, W)

    # Etapa 2: resize reverso para shape original
    recon_resized_back = resize(recon_gray, original_Inorm.shape, order=1, mode='reflect', anti_aliasing=False)

    # Etapa 3: flatten
    recon_flat = recon_resized_back.flatten()
    orig_flat = original_Inorm.flatten()

    # Etapa 4: desnormalizar
    mean_orig = np.mean(orig_flat / r)
    std_orig = np.std(orig_flat / r)
    recon_deno = recon_flat / r
    recon_deno = recon_deno * std_orig + mean_orig

    # Etapa 5: extrair a janela de previsão
    P, total_steps = original_Inorm.shape
    pred_steps = int(0.4 * L * total_steps / (L + H))
    forecast_start = total_steps - pred_steps

    recon_forecast = recon_deno.reshape(P, total_steps)[:, forecast_start:]
    target_forecast = (original_Inorm / r).reshape(P, total_steps)[:, forecast_start:]
    target_forecast = target_forecast * std_orig + mean_orig

    mse = mean_squared_error(target_forecast.flatten(), recon_forecast.flatten())
    mae = mean_absolute_error(target_forecast.flatten(), recon_forecast.flatten())

    print(f"MSE na janela de previsão: {mse:.6f}")
    print(f"MAE na janela de previsão: {mae:.6f}")

    return mse, mae

np.float = float

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


repo_dir = 'mae'
repo_url = 'https://github.com/facebookresearch/mae.git'

if not os.path.exists(repo_dir):
    print(f'Cloned in : {repo_dir}')
    subprocess.run(['git', 'clone', repo_url], check=True)
else:
    print(f'Repository already exists in {repo_dir}')

mae_path = os.path.abspath("mae")
if mae_path not in sys.path:
    sys.path.insert(0, mae_path)

from models_mae import mae_vit_base_patch16
chkpt_dir = 'mae_visualize_vit_base.pth'
model = mae_vit_base_patch16()
checkpoint = torch.load(chkpt_dir, map_location=device)
msg = model.load_state_dict(checkpoint['model'], strict=False)
model = model.to(device)

datasets = {
    "ETTh1": ("https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv", 96),
    "ETTh2": ("https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh2.csv", 96),
    "ETTm1": ("https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm1.csv", 96),
    "ETTm2": ("https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm2.csv", 96)
}

H_values = [96, 192, 336, 720]

results = []

for dataset_name, (url, L) in datasets.items():
    df = pd.read_csv(url)
    attributes = [col for col in df.columns if col not in ['date']]
    for attr in attributes:
        serie = df[attr].values[:512]
        for H in H_values:
            try:
                matriz_2d = serie_to_2Dmatrix(serie)
                Inorm = get_Inorm(matriz_2d)
                Igray_resized_3ch = get_Igrey(Inorm, L, H)

                reconstructed_image = run_one_image_from_aligned_image(
                    Igray_resized_3ch, model, L=L, H=H
                )
                print(f"{dataset_name} | {attr} | L={L}, H={H}")
                mse, mae = reconstruct_and_evaluate_forecast(
                    y_reconstructed=reconstructed_image.numpy(),
                    original_Inorm=Inorm, L=L, H=H
                )

                results.append({
                    'Dataset': dataset_name,
                    'Attribute': attr,
                    'L': L,
                    'H': H,
                    'MSE': mse,
                    'MAE': mae
                })

            except Exception as e:
                print(f"Failed for {dataset_name} - {attr} - L={L}, H={H} -- {e}")


results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
