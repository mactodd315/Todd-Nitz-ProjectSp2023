#!/bin/bash
pycbc_create_injections --verbose \
        --config-files $1 \
        --ninjections $2 \
        --seed 0 \
	--output-file /home/mrtodd/2d_FPES/training_data/injection.hdf \
        --variable-params-section variable_params \
        --static-params-section static_params \
        --dist-section prior \
        --force
