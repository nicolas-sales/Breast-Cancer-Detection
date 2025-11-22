import sys
from src.logger import logging
from src.exception import CustomException

from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.components.test_exporter import TestSetExporter


class TestExporterPipeline:

    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def run(self):
        try:
            logging.info("Starting Test Exporter Pipeline")

            # Récupération données DataIngestionPipeline
            ingestion_pipeline = DataIngestionPipeline(folder_path=self.folder_path)
            data = ingestion_pipeline.run()

            X_test = data["X_test"]
            y_test = data["y_test"]

            exporter = TestSetExporter(output_folder="test_images")
            folder, csv_path = exporter.save_test_set(X_test, y_test)

            logging.info("Test Exporter Pipeline Completed")

            return folder, csv_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    folder = "breast_cancer_public_data/data_2"

    pipeline = TestExporterPipeline(folder_path=folder)
    folder, csv_path = pipeline.run()

    print("Test Exporter Pipeline Successful")
    print("Images folder :", folder)
    print("CSV labels    :", csv_path)