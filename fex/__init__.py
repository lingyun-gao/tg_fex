
# NOTE: model versions in order of recency
all_model_versions = [
    'v2.2017',
    'v1.7tf',
    'v1.7p',
]

latest_model_version = all_model_versions[0]


class FeatureExtractorModel():

    @staticmethod
    def latest(*args, **kwargs):
        return FeatureExtractorModel.version(
            model_version=latest_model_version,
            *args, **kwargs)

    @staticmethod
    def version(model_version, *args, **kwargs):
        if model_version not in all_model_versions:
            raise Exception(
                'Invalid model_version "%s" provided' %
                model_version)

        elif model_version == 'v2.2017':
            from v2_2017 import FeatureExtractorModelVersioned

        elif model_version == 'v1.7tf':
            from v1_7tf import FeatureExtractorModelVersioned

        elif model_version == 'v1.7p':
            from v1_7p import FeatureExtractorModelVersioned

        print('Returning fex model_version: %s' % model_version)
        return FeatureExtractorModelVersioned(*args, **kwargs)

    @staticmethod
    def latest_model_version():
        return latest_model_version

    @staticmethod
    def all_model_versions():
        return all_model_versions


class Transformer():

    @staticmethod
    def version(model_version, *args, **kwargs):
        if model_version not in all_model_versions:
            raise Exception(
                'Invalid model_version "%s" provided' %
                model_version)

        elif model_version == 'v2.2017':
            from v2_2017.transformer import Transformer

        elif model_version == 'v1.7tf':
            from v1_7tf.transformer import Transformer

        elif model_version == 'v1.7p':
            from v1_7p.transformer import Transformer

        return Transformer(*args, **kwargs)
