# About
This is a fork from https://github.com/hikopensource/DAVAR-Lab-OCR with modified setup script 
and added docker web service for LGPMA table extraction model inference.

## Web service setup
Download model `maskrcnn-lgpma-pub-e12-pub.pth` from [Google Drive](https://drive.google.com/drive/folders/1Ik3KCiSATgOlCK4P5TXnbIMBZDOxrv4V) 
and place it in `models` directory.

### Running service locally
* Run `setup.sh` locally and it should install and build all required components.
* Then you can run web service using uvicorn `uvicorn src.web:app --host 0.0.0.0 --port 8000`

### Running service from docker container
* Run `docker build -f docker/Dockerfile -t <tag> .`
* Then you can start the container with web service with `docker run -d -p 8000:8000 -t <tag>`

## Usage
When the web service is running, you can use http protocol to process table images or set
processing device.

### Service endpoints
* `/extract_table_structure/` - returns json with the table data obtained from model.
* `/get_device/` - returns the name of the device used by model, either cpu or cuda.
* `/set_device/{device}/` - tries to set processing device as `device` and returns json with 
currently used device and method output - either success or failure

### Example
```python
import requests

filename = ...
with open(filename, "rb") as f:
    response = requests.post(url="http://0.0.0.0:8000/extract_table_structure", files={"input_file": f})
```

