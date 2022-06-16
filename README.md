# NMLAB-Final
This repo is for security camera on Jetson Nano. 

## Architecture
To be more specific, this repo support the following services.
![image](https://user-images.githubusercontent.com/46078333/174082745-bc8075a3-bda2-41af-8525-019de8f03189.png)

## Human Detection Logic
When receiveing a frame, it will send an alert based on following conditions:
![image](https://user-images.githubusercontent.com/46078333/174083863-a675013d-99dd-48a1-9a2f-81998b2347bb.png)

## Usage
`python3 . --lineID xxx`

## Setup
1. `pip3 install -r requirements.txt`
2. Set up aws key of our service (please contact the author)
