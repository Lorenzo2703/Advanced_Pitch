DLAI project 2023-2024

Alessandro Serafini serafini.1916973@studenti.uniroma1.it
Lorenzo Giare` giare.1886115@studenti.uniroma1.it

The training process depends on the following codes:

train.py
tf_example_deserialization.py
     
To prepare the datasets use the following codes:

download.py 
(or)
{dataset_name}.py

This codes will download the datasets in a directory 'mir_datasets' in your Home directory and will produce the tfrecords needed by train.py.
The produced splits will be at the path '\home\{user}\data\basic_pitch\{dataset_name}\' 
You can use both the following commands to download a dataset:

python download.py --dataset {dataset_name}
python {dataset_name}.py

In order to train the model on the specific dataset simply give the command:

python train.py --{dataset_name}

The codes for evaluation are in the 'experiments' folder

