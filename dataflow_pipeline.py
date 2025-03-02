import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
import json

# Define pipeline options
pipeline_options = PipelineOptions()
pipeline_options.view_as(StandardOptions).streaming = True  # Enable streaming

# Set the input Pub/Sub topic
input_topic = "projects/genuine-arena-448912-v1/topics/MS3-input"
output_topic = "projects/genuine-arena-448912-v1/topics/MS3-output"

class ProcessMessage(beam.DoFn):
    """A DoFn to process each message, detect pedestrian, and estimate depth."""
    def process(self, element):
        message = json.loads(element.decode('utf-8'))  # Decode Pub/Sub message
        result = {
            "bounding_boxes": [[10, 20, 30, 40]],  # Example bounding box
            "depth": 5.6  # Example depth estimation
        }
        yield json.dumps(result).encode('utf-8')

# Define the Beam pipeline
with beam.Pipeline(options=pipeline_options) as p:
    (
        p
        | "ReadFromPubSub" >> beam.io.ReadFromPubSub(topic=input_topic)
        | "ProcessMessages" >> beam.ParDo(ProcessMessage())
        | "WriteToPubSub" >> beam.io.WriteToPubSub(topic=output_topic)
    )
