import tensorflow as tf
from tensorflow.core.util import event_pb2
from pathlib import Path

train_metrics = [
    "Loss/classification_loss",
    "Loss/localization_loss",
    "Loss/normalized_total_loss",
    "Loss/regularization_loss",
    "Loss/total_loss",
    "learning_rate",
]

event_files = [str(f) for f in Path('./logs/train').rglob('events.out.*')]

for event_file in event_files:
    serialized_examples = tf.data.TFRecordDataset(event_file)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for value in event.summary.value:
            if value.tag in train_metrics:
              t = tf.make_ndarray(value.tensor)
              print(value.tag, event.step, t, type(t))