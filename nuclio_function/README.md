# Nuclio Serverless Function for AOC Fruit Detector

This directory contains the implementation of a nuclio serverless function that wraps the AOC fruit detector for use with CVAT (Computer Vision Annotation Tool).

## Architecture

The solution consists of two containers:

1. **Fruit Detector Container**: The existing ROS2 node that performs fruit detection using Detectron2
2. **Nuclio Function Container**: A serverless function that bridges between CVAT and the fruit detector

## Communication Flow

1. CVAT sends image via HTTP POST to nuclio function
2. Nuclio function publishes image to ROS2 topic `/camera/image_raw`
3. Fruit detector ROS2 node processes the image and publishes results to `/fruit_info` topic
4. Nuclio function subscribes to `/fruit_info` topic and receives detection results
5. Nuclio function converts results to CVAT annotation format and returns via HTTP response

## Files

- `main.py`: Main nuclio function handler
- `ros2_bridge.py`: ROS2 communication bridge
- `cvat_converter.py`: Converts fruit detection results to CVAT format
- `Dockerfile`: Container definition for nuclio deployment
- `function.yaml`: Nuclio function configuration
- `requirements.txt`: Python dependencies
- `deploy.sh`: Deployment script
- `test_function.py`: Test script for validating the integration

## Prerequisites

1. **Nuclio**: Install nuclio CLI and have a Kubernetes cluster or Docker environment
   - Follow: https://docs.nuclio.io/en/stable/setup/k8s/getting-started-k8s/

2. **AOC Fruit Detector**: The fruit detector ROS2 node should be running and accessible
   - Ensure the node is publishing to `/fruit_info` topic
   - Ensure the node is subscribing to `/camera/image_raw` topic

3. **ROS2 Network**: Both containers must be on the same ROS2 network/domain

## Deployment

### Method 1: Using the deployment script

```bash
cd nuclio_function
./deploy.sh
```

### Method 2: Manual deployment

1. **Build the Docker image:**
   ```bash
   docker build -t aoc-fruit-detector:latest .
   ```

2. **Deploy to nuclio:**
   ```bash
   nuctl deploy aoc-fruit-detector \
     --namespace nuclio \
     --path . \
     --file function.yaml \
     --registry "" \
     --run-image aoc-fruit-detector:latest
   ```

3. **Check deployment status:**
   ```bash
   nuctl get function aoc-fruit-detector --namespace nuclio
   ```

## Configuration

### Environment Variables

- `ROS_DOMAIN_ID`: ROS2 domain ID (default: 0)
- `RMW_IMPLEMENTATION`: ROS2 middleware implementation (default: rmw_cyclonedx_cpp)

### Function Parameters

The function accepts the following parameters in the HTTP request:

```json
{
  "image": "<base64_encoded_image>",
  "threshold": 0.5
}
```

- `image`: Base64 encoded image data (required)
- `threshold`: Confidence threshold for detections (optional, default: 0.5)

## Testing

### Using the test script

```bash
python3 test_function.py \
  --url http://<function-endpoint> \
  --image /path/to/test/image.jpg \
  --threshold 0.7 \
  --output /path/to/output/annotated_image.jpg
```

### Manual testing with curl

```bash
# Encode image to base64
IMAGE_B64=$(base64 -w 0 /path/to/image.jpg)

# Send request
curl -X POST \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_B64\", \"threshold\": 0.5}" \
  http://<function-endpoint>
```

## CVAT Integration

### 1. Deploy the nuclio function

Follow the deployment instructions above.

### 2. Configure CVAT

1. In CVAT, go to Models → Add Model
2. Set the model type to "Detector"
3. Enter the nuclio function URL
4. Configure labels according to your fruit types (strawberry, tomato, etc.)

### 3. Use for annotation

1. Create or open a task in CVAT
2. Go to Actions → Automatic Annotation
3. Select the AOC Fruit Detector model
4. Configure parameters and run

## Expected Output Format

The function returns CVAT-compatible annotations:

```json
[
  {
    "confidence": 0.95,
    "label": "strawberry_ripe",
    "points": [x1, y1, x2, y2],
    "type": "rectangle",
    "attributes": {
      "variety": "unknown",
      "quality": "High",
      "ripeness_category": "Ripe",
      "ripeness_level": 0.85,
      "area": 1234.5,
      "occlusion_level": 0.1
    }
  }
]
```

## Troubleshooting

### Common Issues

1. **ROS2 Connection Failed**: Ensure both containers are on the same ROS2 domain
2. **Timeout Errors**: Check if the fruit detector node is running and responsive
3. **No Detections**: Verify confidence threshold and image quality
4. **Import Errors**: Ensure all ROS2 packages are properly built and sourced

### Debugging

1. **Check function logs:**
   ```bash
   nuctl logs aoc-fruit-detector --namespace nuclio
   ```

2. **Verify ROS2 topics:**
   ```bash
   ros2 topic list
   ros2 topic echo /fruit_info
   ```

3. **Test fruit detector directly:**
   ```bash
   ros2 launch aoc_fruit_detector fruit_detection.launch.py
   ```

## Development

For development and testing, you can run the components locally:

1. **Start the fruit detector:**
   ```bash
   ros2 launch aoc_fruit_detector fruit_detection.launch.py
   ```

2. **Run the nuclio function locally:**
   ```bash
   cd nuclio_function
   python3 main.py
   ```

## References

- [Nuclio Documentation](https://docs.nuclio.io/en/stable/)
- [CVAT Serverless Tutorial](https://docs.cvat.ai/docs/manual/advanced/serverless-tutorial/)
- [Deploying Functions from Dockerfile](https://docs.nuclio.io/en/stable/tasks/deploy-functions-from-dockerfile.html)