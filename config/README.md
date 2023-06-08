# Configuration file

## The YAML schema

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
        
        [backend]: <backend_for_pretrained_model>, # OPTIONAL, the backend to use for the pretrained model [5]
        
        [task]: <pretrained_model_task>, # OPTIONAL, only for textual modality [6]
      }
    
      ...
    
    ]
  
  ... 

...

```

## Notes
Please refer to the \[*\] reported in the YAML schema from above.

**\[1\]** In case of textual/items, the tsv input file is supposed to be formatted in the following manner:
```
<ITEM_ID_COLUMN_NAME>\t<ITEM_DESCRIPTION_COLUMN_NAME>
[first_item_id]\t[first_item_description]
...
[last_item_id]\t[last_item_description]
```
where <ITEM_ID_COLUMN_NAME> and <ITEM_DESCRIPTION_COLUMN_NAME> are customizable. Note that if no ```item_column``` is provided in the configuration file, Ducho takes the last column (i.e., <ITEM_DESCRIPTION_COLUMN_NAME>) of the tsv file as item column by default.

**\[2\]** In case of textual/interactions, the tsv input file is supposed to be formatted in the following manner:
```
<USER_ID_COLUMN_NAME>\t<ITEM_ID_COLUMN_NAME>\t<REVIEW_COLUMN_NAME>
[first_user_id]\t[first_item_id]\t[first_review]
...
[last_user_id]\t[last_item_id]\t[last_review]
```
where <USER_ID_COLUMN_NAME>, <ITEM_ID_COLUMN_NAME>, and <REVIEW_COLUMN_NAME> are customizable. Note that if no ```interaction_column``` is provided in the configuration file, Ducho takes the last column (i.e., <REVIEW_COLUMN_NAME>) of the tsv file as interaction column by default.

**\[3\]** We provide a modality/backend table for this:

<table>
<thead>
  <tr>
    <th></th>
    <th>Audio<br></th>
    <th>Video</th>
    <th>Textual</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Tensorflow</td>
    <td></td>
    <td>Visual models</td>
    <td></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>TorchAudio</td>
    <td>TorchVideo</td>
    <td></td>
  </tr>
  <tr>
    <td>Transformers</td>
    <td>Transformers</td>
    <td></td>
    <td>Transformers & Sentence</td>
  </tr>
</tbody>
</table>


**\[4\]** We provide a modality/backend table for this:

**\[5\]** We provide a modality/backend table for this:

**\[6\]** The list of available tasks is [here](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.task).

