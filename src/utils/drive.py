"""
Sincronização com o Google Drive.
Cobre download por file_id (dados do dataset) e upload/download
de checkpoints por filename dentro de uma pasta.

Ativado quando drive.enabled=true no config.yaml.
"""
import io
import os
from pathlib import Path


class DriveSync:
    """
    Wrapper unificado para o Google Drive API v3.

    Responsabilidades:
      - download_by_id  : baixa qualquer arquivo pelo seu file_id (dataset, GT, ZIP)
      - upload          : envia checkpoint para a pasta de checkpoints
      - download        : recupera checkpoint pelo nome dentro da pasta
    """

    def __init__(self, service, folder_id: str) -> None:
        self.service = service
        self.folder_id = folder_id

    # ------------------------------------------------------------------
    # Download por file_id direto (dataset)
    # ------------------------------------------------------------------

    def download_by_id(
        self,
        file_id: str,
        dest_path: str | Path,
        show_progress: bool = True,
    ) -> bool:
        """
        Baixa um arquivo do Drive pelo seu file_id.
        Usado para JSON de anotações, ZIP do dataset e GT de Cobb.

        Args:
            file_id      : ID do arquivo no Google Drive
            dest_path    : caminho local de destino
            show_progress: exibe % de progresso no terminal

        Returns:
            True se o download foi concluído com sucesso.
        """
        from googleapiclient.http import MediaIoBaseDownload

        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            request = self.service.files().get_media(fileId=file_id)
            with io.FileIO(str(dest_path), "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    progress, done = downloader.next_chunk()
                    if show_progress and progress:
                        pct = int(progress.progress() * 100)
                        print(f"  {pct}%", end="\r")
            if show_progress:
                print(f"  100% — {dest_path.name}")
            return True
        except Exception as e:
            print(f"Erro ao baixar {file_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Upload/download de checkpoints (por nome dentro da pasta)
    # ------------------------------------------------------------------

    def _get_file_id_by_name(self, filename: str) -> str | None:
        """Localiza um arquivo pelo nome dentro da pasta de checkpoints."""
        query = (
            f"name = '{filename}' and '{self.folder_id}' in parents "
            f"and trashed = false"
        )
        results = (
            self.service.files()
            .list(q=query, fields="files(id)")
            .execute()
        )
        files = results.get("files", [])
        return files[0]["id"] if files else None

    def upload(self, local_path: Path, drive_filename: str) -> None:
        """Faz upload de um checkpoint para a pasta do Drive."""
        from googleapiclient.http import MediaFileUpload

        file_id = self._get_file_id_by_name(drive_filename)
        media = MediaFileUpload(str(local_path), resumable=True)
        if file_id:
            self.service.files().update(
                fileId=file_id, media_body=media
            ).execute()
        else:
            metadata = {"name": drive_filename, "parents": [self.folder_id]}
            self.service.files().create(
                body=metadata, media_body=media
            ).execute()

    def download(self, drive_filename: str, local_path: Path) -> bool:
        """Recupera um checkpoint pelo nome dentro da pasta do Drive."""
        file_id = self._get_file_id_by_name(drive_filename)
        if not file_id:
            return False
        return self.download_by_id(file_id, local_path, show_progress=False)


# ---------------------------------------------------------------------------
# Helpers de conveniência para uso no notebook
# ---------------------------------------------------------------------------

def download_dataset(
    drive_service,
    json_file_id: str,
    json_test_file_id: str,
    zip_file_id: str,
    cobb_gt_train_file_id: str,
    cobb_gt_test_file_id: str,
    local_data_path: str = "/content/dataset",
) -> dict:
    """
    Baixa todos os arquivos do dataset (JSON, ZIP, GT treino e GT teste)
    se ainda não existirem localmente, e descompacta o ZIP.

    Args:
        drive_service         : serviço autenticado do Google Drive API v3
        json_file_id          : ID do Spinal-AI2024_train_annotation.json
        json_test_file_id     : ID do Spinal-AI2024_test_annotation.json
        zip_file_id           : ID do Spinal-AI2024-dataset.zip
        cobb_gt_train_file_id : ID do Spinal-AI2024_Cobb_train_gt.txt
        cobb_gt_test_file_id  : ID do Spinal-AI2024_Cobb_test_gt.txt

    Returns:
        Dicionário com todos os caminhos locais prontos para uso.
    """
    os.makedirs(local_data_path, exist_ok=True)

    paths = {
        "json":         os.path.join(local_data_path, "Spinal-AI2024_train_annotation.json"),
        "json_test":    os.path.join(local_data_path, "Spinal-AI2024_test_annotation.json"),
        "zip":          os.path.join(local_data_path, "Spinal-AI2024-dataset.zip"),
        "cobb_gt_train": os.path.join(local_data_path, "Spinal-AI2024_Cobb_train_gt.txt"),
        "cobb_gt_test":  os.path.join(local_data_path, "Spinal-AI2024_Cobb_test_gt.txt"),
        "images_root":  os.path.join(local_data_path, "Spinal-AI2024-dataset"),
    }

    sync = DriveSync(drive_service, folder_id="")

    if not os.path.exists(paths["json"]):
        print("Baixando annotation JSON...")
        sync.download_by_id(json_file_id, paths["json"])

    if not os.path.exists(paths["json_test"]):
        print("Baixando annotation JSON (teste)...")
        sync.download_by_id(json_test_file_id, paths["json_test"])

    if not os.path.exists(paths["cobb_gt_train"]):
        print("Baixando Cobb GT (treino)...")
        sync.download_by_id(cobb_gt_train_file_id, paths["cobb_gt_train"])

    if not os.path.exists(paths["cobb_gt_test"]):
        print("Baixando Cobb GT (teste)...")
        sync.download_by_id(cobb_gt_test_file_id, paths["cobb_gt_test"])

    if not os.path.exists(paths["zip"]):
        print("Baixando dataset ZIP...")
        sync.download_by_id(zip_file_id, paths["zip"])

    if not os.path.exists(paths["images_root"]):
        print("Descompactando dataset...")
        os.system(f'unzip -q "{paths["zip"]}" -d "{paths["images_root"]}"')

    print("Dataset pronto.")
    return paths