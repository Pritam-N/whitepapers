from google.cloud import aiplatform
import config
import logging
 
def create_vertex_dataset_text_semantic():
    """This function creates the datset of type text
    which can be used for model training.
    """
    aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
 
    dataset_created = aiplatform.TextDataset.create(
        display_name=config.DATASET_DISPLAY_NAME,
        gcs_source=config.GCS_LOCATION + config.REVIEWS_CSV_FILE,
        import_schema_uri=aiplatform.schema.dataset.ioformat.text.single_label_classification,
    )
    dataset_created.wait()
    logging.info(f'Dataset resouce name is {dataset_created.resource_name}.')