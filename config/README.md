# Configuration file

Here we report a schematic representation of a generic YAML file for the configuration of any multimodal feature extraction.

```yaml
dataset_path: <path_to_the_dataset_folder> # either relative or absolute path 

gpu_list: <list_of_gpu_ids> # list of gpu ids to use during the extraction, -1 for cpu computation

[visual|textual|audio]: 
  [items|interactions]:
    input_path: <input_file_or_folder> # this path is relative to dataset_path
    output_path: <output_folder> # this path is relative to dataset_path
    model: [
      {
        name: <model_name>, # as indicated in the specific backend you are using
        output_layers: <list_of_output_layers>, # 
      ...
    ]
```
