# Graph-Distribution-Learning

visualize_data.py contains functions display_graph_stats and display_label_stats which are used to visualize the data generically or to visualize the label distributions of the data. To use either, create the dataset as a geo.data.Data() object, then pass it into the function. display_label_stats will also need an additional string that denotes the dataset's name.

grid_search_dynamic.py provides a demo template for hyperparameter search of GLDL dynamic setting.
single_run.py provides a demo for a single training case for GLDL.
1. To reproduce the results in the paper, we have trained the model and saved the trained model. The trained model state is in the ./src/trained_model/ directory in a zip format. Please unzip the file and put them under the trained_model directory first.
2. set up the environment following the requirements.txt
3. For different datasets, run the following commands:
ACM:\
GCN:
~~~
python over_smothing.py --dataset acm --p 50 --hidden_channel 128 --num_layers 3 --patience 10 --lr 0.0001 --weight_decay 0.0005
~~~
static:
~~~
python quick_run.py --dataset acm --p 50 --lr 0.0001 --weight_decay 0.0005 --gcnhidden 320 --pgehidden 64 --freql 30 --freqv 50 --gcnvlayers 2 --gcnllayers 2 --pgelayers 3 --mode static --epochs 500 --patience 50
~~~
dynamic:
~~~
python quick_run.py --dataset acm --p 50 --patience 50 --epochs 500 --mode dynamic --inner_epochs 70 --gcnhidden 450 --pgehidden 64 --freql 50 --freqv 30 --gcnvlayers 2 --gcnllayers 2 --pgelayers 3 --lr 0.0001 --weight_decay 0.0005
~~~
DBLP\
GCN:
~~~
python over_smothing.py --dataset dblp --p 50 --hidden_channel 32 --num_layers 2 --patience 10 --lr 0.001 --weight_decay 0.0005
~~~
static : 
~~~
python quick_run.py --dataset dblp --p 50 --mode static --gcnhidden 200 --freql 30 --freqv 80 --gcnvlayers 2 --gcnllayers 1 --lr 0.0001 --weight_decay 0.0005 --epochs 500 --patience 100
~~~
dynamic:
~~~
python quick_run.py --dataset dblp --p 50 --gcnhidden 300 --pgehidden 64 --freql 30 --freqv 20 --lr 0.0001 --weight_decay 0.0005 --gcnvlayers 2 --gcnllayers 2 --pgelayers 2 --mode dynamic --save 1 --epochs 500 --patience 20
~~~
Yelp\
GCN:
~~~
python over_smothing.py --dataset yelp2 --p 50 --hidden_channel 32 --num_layers 3 --patience 50 --lr 0.0001 --weight_decay 0.005
~~~
static: 
~~~
python quick_run.py --dataset yelp --p 50 --gcnhidden 64 --gcnvlayers 2 --gcnllayers 2 --freqv 20 --freql 60 --lr 0.0001 --weight_decay 0 --mode static --patience 20 --epochs 500
~~~
dynamic: 
~~~
python quick_run.py --dataset yelp --p 50 --gcnhidden 64 --pgehidden 32 --freql 60 --freqv 20 --lr 0.0001 --weight_decay 0.0005 --gcnvlayers 4 --gcnllayers 2 --pgelayers 2 --mode dynamic --epochs 500 --patience 20 --inner_epochs 50
~~~
Yelp2\
GCN: 
~~~
python over_smothing.py --dataset yelp2 --p 50 --hidden_channel 128 --num_layers 3 --patience 50 --lr 0.0001 --weight_decay 0.0005 --max_epochs 1000
~~~
static:
~~~
python quick_run.py --dataset yelp2 --p 50 --gcnhidden 256 --pgehidden 64 --freql 60 --freqv 60 --lr 0.001 --weight_decay 0.0005 --gcnvlayers 4 --gcnllayers 1 --pgelayers 2 --mode static --epochs 500 --patience 50 --inner_epochs 50
~~~
dynamic: 
~~~
python quick_run.py --dataset yelp2 --p 50 --gcnhidden 320 --pgehidden 64 --freql 30 --freqv 60 --lr 0.0001 --weight_decay 0.0005 --gcnvlayers 3 --gcnllayers 2 --pgelayers 2 --mode dynamic --epochs 500 --patience 50 --inner_epochs 50
~~~


