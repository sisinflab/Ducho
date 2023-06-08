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
        
        [backend]: <backend_for_pretrained_model>, # OPTIONAL, the backend to use for the pretrained model [3]
        
        [task]: <pretrained_model_task>, # OPTIONAL, only for textual modality [5]
      }
    
      ...
    
    ]
  
  ... 

...

```

## Notes
Please refer to the \[*\] reported in the YAML schema from above.

**\[1\]** In case of textual/items, the tsv input file is supposed to be formatted in the following manner:

```tsv
<ITEM_ID_COLUMN_NAME>\t<ITEM_DESCRIPTION_COLUMN_NAME>
[first_item_id]\t[first_item_description]
...
[last_item_id]\t[last_item_description]
```
where <ITEM_ID_COLUMN_NAME> and <ITEM_DESCRIPTION_COLUMN_NAME> are customizable. Note that if no ```item_column``` is provided in the configuration file, Ducho takes the last column (i.e., <ITEM_DESCRIPTION_COLUMN_NAME>) of the tsv file as item column by default.

**\[2\]** In case of textual/interactions, the tsv input file is supposed to be formatted in the following manner:

```tsv
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
    <td> <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications#modules_2">link</a> </td>
    <td></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td> <a href="https://pytorch.org/audio/stable/pipelines.html#module-torchaudio.pipelines">link</a> </td>
    <td> <a href="https://pytorch.org/vision/stable/models.html">link</a> </td>
    <td></td>
  </tr>
  <tr>
    <td>Transformers</td>
    <td> <a href="https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Model">link</a> </td>
    <td></td>
    <td>Transformers: <a href="https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline">link</a><br>SentenceTransformers: <a href="https://www.sbert.net/docs/pretrained_models.html#model-overview">link</a></td>
  </tr>
</tbody>
</table>


**\[4\]** Depending on the backend you are using:
- **TensorFlow**: use the **exact** same naming scheme obtained by calling the method ```summary()``` on the instantiated model object. For example:
```python
import tensorflow
resnet50 = getattr(tensorflow.keras.applications, 'ResNet50')()
print(resnet50.summary())

"""
here is the final part of the console output:
...
conv5_block3_add (Add)         (None, 7, 7, 2048)   0           ['conv5_block2_out[0][0]',       
                                                                  'conv5_block3_3_bn[0][0]']      
                                                                                                  
 conv5_block3_out (Activation)  (None, 7, 7, 2048)   0           ['conv5_block3_add[0][0]']       
                                                                                                  
 avg_pool (GlobalAveragePooling  (None, 2048)        0           ['conv5_block3_out[0][0]']       
 2D)                                                                                              
                                                                                                  
 predictions (Dense)            (None, 1000)         2049000     ['avg_pool[0][0]']               
                                                                                                  
==================================================================================================
Total params: 25,636,712
Trainable params: 25,583,592
Non-trainable params: 53,120
__________________________________________________________________________________________________

in this case, for example, 'avg_pool' is what we are looking for.

"""

```
- **PyTorch+Textual:** indicate the minimum path to reach the output layer (using the **exact** same names obtained by instantiating the model object and separated by '.'). For example:

```python
import torchvision
alexnet = getattr(torchvision.models, 'alexnet')(weights='DEFAULT')
print(alexnet)

"""
here is the console output:
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

in this case, for example, 'classifier.3' is what you are looking for.
"""
```

- **PyTorch+Audio:** depending on the pre-trained model, you may be asked to indicate the layer **number** in ascending (i.e., \[0, L-1\]) or descending order (i.e., \[L-1, 0\]). Once again, just instantiate the model and see its structure to find it out.

- **Transformers+Textual:** you are asked to indicate the layer **number** in descending order (i.e., \[L-1, 0\]). In the case of SentenceTransformers, you don't really need to indicate any output layer (the backend already comes with its own extraction which is fixed).

- **Transformers+Audio:** you are asked to indicate the layer **number** in ascending order (i.e., \[L-1, 0\]).

**\[5\]** The list of available tasks is [here](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.task).

