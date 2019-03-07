import FeatureExtractor, Transformer

USE_GPU = False
BATCH_SIZE = 8

t = Transformer(batch_size=BATCH_SIZE)

# TODO: reverting back to Theano until new TF models are tested
# m = FeatureExtractor(net_batch_size=BATCH_SIZE)
m = FeatureExtractor(net_batch_size=BATCH_SIZE, use_gpu=USE_GPU)


transforms = t.transform_blobs(blobs, crops_filtered)
time_delta = time.time() - time0
self.logger.info('Took %0.4f seconds to transform %d images: %0.2f images/s' \
                 % (time_delta, len(batch), len(batch) * 1.0 / time_delta))

self.logger.debug("Placing batch into embed queue:")
self.embed_q.put((self.priority, (batch_filtered, transforms)))



forwards = self.model.forward(transforms)

for i, (task_crop, forward) in enumerate(zip(batch, forwards)):
    ''' Example:

        (Task(id='abc'), [0.1,0.1,0.9,0.9]), {'embedding':..., 'tags': ...}
    '''
    task, crop = task_crop

    if i == 0:
        cur_task = task

    if task.id != cur_task.id:
        _complete_task(cur_task)
        cur_task = task

    embedding, tags = None, None

    if forward:
        embedding = forward['embedding']
        tags = forward['tags']