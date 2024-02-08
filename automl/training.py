import kfp
from google.cloud import aiplatform as aip
from google_cloud_pipeline_components.v1.dataset import GetVertexDatasetOp
from google_cloud_pipeline_components.v1.automl.training_job import AutoMLTextTrainingJobRunOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from kfp.dsl import Input , Artifact
 
import os
import config
 
pipeline_root_path = os.path.join(
        config.GCS_LOCATION, 'pipeline_root',
        config.PIPELINE_NAME
        )
 
@kfp.dsl.component(packages_to_install=["google-cloud-aiplatform"],base_image='python:3.9')
def sentiments_model_evaluation(
    model: Input[Artifact]
    ):
    """ This custom component is used to evaluate the performance of model.
    Args:
        model: Model trained in Trainer Component
    """
    from google.cloud import aiplatform
    model_resource_path = model.metadata["resourceName"]
 
    # Get the trained model resource
    model = aiplatform.Model(model_resource_path)
    evaluations= model.list_model_evaluations()
    for eval in evaluations:
        print("f1-score", eval.metrics["f1Score"])
 
 
# Define the workflow of the pipeline.
@kfp.dsl.pipeline(
    name="automl-semantic-analysis-training-v2",
    pipeline_root=pipeline_root_path)
def reviews_semantic_analysis():
    """This creates the kubeflow pipeline which performs loads dataset ,
    training model and upload same for predictions.
    """
 
    dataset_name = 'projects/' + config.PROJECT_ID + '/locations/' + config.LOCATION + '/datasets/' + config.DATASET_ID
    ds_op = GetVertexDatasetOp(
        dataset_resource_name=dataset_name)
 
    # This step is a model training component. It takes the dataset output from
    # the first step, supplies it as an input argument to the component.
    training_job_run_op = AutoMLTextTrainingJobRunOp(
        project=config.PROJECT_ID,
        display_name=config.TRAINING_MODEL_DISPLAY_NAME,
        prediction_type="sentiment",
        dataset=ds_op.outputs["dataset"],
        model_display_name=config.TRAINING_MODEL_DISPLAY_NAME,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        sentiment_max=1
    )
    sentiments_model_evaluation(model=training_job_run_op.outputs["model"])
 
    # create endpoint and deploy the model
    create_endpoint_op = EndpointCreateOp(
        project=config.PROJECT_ID,
        display_name="endpoint"
        )
 
    ModelDeployOp(
        model=training_job_run_op.outputs["model"],
        endpoint=create_endpoint_op.outputs['endpoint'],
        automatic_resources_min_replica_count=1,
        automatic_resources_max_replica_count=1
        )
 
 
#  complie the kubeflow pipeline and run the job
kfp.compiler.Compiler().compile(
    pipeline_func=reviews_semantic_analysis,
    package_path='reviews_analysis.yaml'
)
 
 
aip.init(
    project=config.PROJECT_ID,
    location=config.LOCATION,
)
 
# Prepare the pipeline job
job = aip.PipelineJob(
    display_name="automl",
    template_path="reviews_analysis.yaml",
    pipeline_root=pipeline_root_path
)
 
job.submit()