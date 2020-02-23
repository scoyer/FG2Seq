#!/bin/bash

#InCar Dataset
echo "Preprocessing InCar dataset..."
for data in train dev test
do
    python data/KVR/read_data.py --json="data/KVR/kvret_${data}_public.json" --entity="data/KVR/kvret_entities.json" > "data/KVR/${data}.txt"
done

#CamRest Dataset
echo "Preprocessing CamRest dataset..."
for data in train dev test
do
    python data/CamRest/read_data.py --json="data/CamRest/CamRest676_${data}.json" > "data/CamRest/${data}.txt"
done

