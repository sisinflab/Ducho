# Configuration file

Here we report a schematic representation of a generic YAML file for the configuration of any multimodal feature extraction.

```yaml
dataset_path: <path_to_the_dataset_folder> # either relative or absolute path 

gpu_list: <list_of_gpu_ids> # list of gpu ids to use during the extraction, -1 for cpu computation

visual|textual|audio: 

  items|interactions:
  
    input_path: <input_file_or_folder> # this path is relative to dataset_path
    
    output_path: <output_folder> # this path is relative to dataset_path
    
    [item_column]: <column_for_item_descriptions> # OPTIONAL, the column name for the item description in the tsv file [1]
    
    [interaction_column]: <column_for_interaction_reviews> # OPTIONAL, the column name for the interaction reviews in the tsv file [2]
    
    model: [
      {
        name: <model_name>, # as indicated in the specific backend you are using [3]
        
        output_layers: <list_of_output_layers>, # as indicated in the specific backend you are using [4]
        
        [reshape]: <reshape_size>, # OPTIONAL, a tuple only for visual modality
        
        [clear_text]: <whether_to_clear_input_text>, # OPTIONAL, a boolean for textual modality
        
        [framework]: <backend_for_pretrained_model>, # OPTIONAL, the backend to use for the pretrained model [5]
        
        [task]: <pretrained_model_task>, # OPTIONAL, only for textual modality
      }
    
      ...
    
    ]
  
  ... 

...

```
