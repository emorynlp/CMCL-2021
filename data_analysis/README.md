# Requirements
```
pip install -r requirement.txt
```

# Emotion Wheel Generation
```sh
$ python generate_emotion_positions.py emb_avg_3layer_avg/EDQA_none.norm.emb.run
```
The output will be a table including 4 columns of `non-basic emotion`, `basic emotion pair`, `position`, and `score` as below.
```
caring		trusting_sad		4	0.28
devastated	surprised_sad		8	0.93
annoyed		angry_anticipating	0	0.80
hopeful		anticipating_trusting	1	0.67
```
Note the `position` is a integer from 0 to 9, which stands for the distance between the non-basic emotion and the first basic emotion of the pair on the circle. For example, `devastated` is at the position between `surprised` and `sad` and very close to `sad`.


# PAD Regression Model
Three regression models are trained independently. For example, the following command is to train the rergession model for `pleasure(p)`, and the maximum epoch is set as `10`.
```sh
$ python pytorch_mlp.py p 10
```

The output would be as below:
```
len of em_with_pad: 22
Epoch [1/10], Step [1], Loss: 0.32272592186927795
Epoch [2/10], Step [1], Loss: 0.22282899916172028
Epoch [3/10], Step [1], Loss: 0.1426474004983902
Epoch [4/10], Step [1], Loss: 0.07973825186491013
Epoch [5/10], Step [1], Loss: 0.059601135551929474
Epoch [6/10], Step [1], Loss: 0.047198884189128876
Epoch [7/10], Step [1], Loss: 0.03421207144856453
Epoch [8/10], Step [1], Loss: 0.028345981612801552
Early Stop: Epoch [9/10], Step [1], Loss: 0.034115225076675415
caring	0.54
devastated	-0.47
trusting	0.17
annoyed	-0.42
hopeful	0.51
...
```

It shows that the best epoch should be `8`, so we run the command again using the parameter `8` to get the best results.
```sh
$ python pytorch_mlp.py p 8
```

The results are:
```sh
len of em_with_pad: 22
Epoch [1/8], Step [1], Loss: 0.32272592186927795
Epoch [2/8], Step [1], Loss: 0.22282899916172028
Epoch [3/8], Step [1], Loss: 0.1426474004983902
Epoch [4/8], Step [1], Loss: 0.07973825186491013
Epoch [5/8], Step [1], Loss: 0.059601135551929474
Epoch [6/8], Step [1], Loss: 0.047198884189128876
Epoch [7/8], Step [1], Loss: 0.03421207144856453
Epoch [8/8], Step [1], Loss: 0.028345981612801552
caring	0.46
devastated	-0.41
trusting	0.07
annoyed	-0.56
hopeful	0.55
...
```

# Layer-wise Analysis
TBD