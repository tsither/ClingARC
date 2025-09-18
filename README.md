# ClingARC
A neuro-symbolic framework using Answer Set Programming and LLMs to tackle the ARC-AGI 1 dataset. 


### Run an instance
You can run ClingARC on a single ARC task via 'main.py'. 

```bash
python main.py --mode [wholesale,iterative] --instance [path to instance (e.g. instances/1]
```

*Note, the framework has only been tested on a subset of the larger ARC dataset, behavior could be unpredictable or unreliable on instances where the grid size is variable from input to output. 


### Converting ARC JSON data into ClingARC
ClingARC requires ARC data to be translated into equivalent Answer Set Programming fact format. You can automatically process and convert an ARC directory using 'prepare_data.py'.

```bash
python prepare_data.py --arc-data [directory storing ARC data] --output-dir [desired output directory name]
```
