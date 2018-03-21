# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

from classifier.model_classifier import classifier

def main():

    # Select experiment below

    from classifier.experiments import synthetic_normalnet as exp_config
    # from classifier.experiments import adni_experiment as exp_config

    # Get Data
    if exp_config.data_identifier == 'synthetic':
        from data.synthetic_data import synthetic_data as data_loader
    elif exp_config.data_identifier == 'adni':
        from data.adni_data import adni_data as data_loader
    else:
        raise ValueError('Unknown data identifier: %s' % exp_config.data_identifier)

    data = data_loader(exp_config)

    # Build VAGAN model
    classifier_model = classifier(exp_config=exp_config, data=data, fixed_batch_size=exp_config.batch_size)

    # Train VAGAN model
    classifier_model.train()


if __name__ == '__main__':

    main()
