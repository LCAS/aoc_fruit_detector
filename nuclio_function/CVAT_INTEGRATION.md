# CVAT Integration Example

This document provides an example of how to integrate the AOC Fruit Detector nuclio function with CVAT.

## Prerequisites

1. CVAT is installed and running
2. Nuclio is installed in your Kubernetes cluster or Docker environment
3. AOC Fruit Detector nuclio function is deployed

## Step 1: Deploy the Function

```bash
cd nuclio_function
./deploy.sh
```

Note the function endpoint URL from the deployment output.

## Step 2: Configure CVAT Model

1. **Access CVAT Admin Interface**
   - Go to CVAT admin panel (usually at `http://your-cvat-url/admin`)
   - Login with admin credentials

2. **Add New Model**
   - Navigate to `Models` → `Add Model`
   - Fill in the following details:

   ```
   Name: AOC Fruit Detector
   Owner: <your-username>
   URL: http://<nuclio-function-endpoint>
   Labels: strawberry,tomato,apple,pear
   Model type: Detector
   Storage method: Local
   Enabled: ✓
   ```

3. **Configure Labels**
   Create labels that match your fruit detection types:
   ```
   - strawberry
   - strawberry_ripe
   - strawberry_unripe
   - strawberry_overripe
   - tomato
   - tomato_ripe
   - tomato_unripe
   - tomato_overripe
   ```

## Step 3: Use in CVAT Tasks

1. **Create or Open Task**
   - Create a new task or open an existing one
   - Upload images containing fruits

2. **Run Automatic Annotation**
   - Go to `Actions` → `Automatic Annotation`
   - Select "AOC Fruit Detector" from the model dropdown
   - Configure parameters:
     ```
     Threshold: 0.5 (adjust based on your needs)
     Maximum annotations per frame: 100
     ```
   - Click "Submit"

3. **Review Results**
   - The function will process each frame
   - Review and adjust annotations as needed
   - Use CVAT's annotation tools to refine results

## Example API Request

For direct testing, you can send requests to the nuclio function:

```bash
# Test with a sample image
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -w 0 /path/to/fruit_image.jpg)'",
    "threshold": 0.5
  }' \
  http://<nuclio-function-endpoint>
```

Expected response:
```json
[
  {
    "confidence": 0.95,
    "label": "strawberry_ripe",
    "points": [123.4, 56.7, 234.5, 167.8],
    "type": "rectangle",
    "attributes": {
      "variety": "unknown",
      "quality": "High",
      "ripeness_category": "Ripe",
      "ripeness_level": 0.85
    }
  }
]
```

## Troubleshooting

### Common Issues

1. **Function Not Responding**
   - Check if nuclio function is running: `nuctl get functions`
   - Check function logs: `nuctl logs aoc-fruit-detector`

2. **No Detections Returned**
   - Lower the confidence threshold
   - Check if fruit detector ROS2 node is running
   - Verify ROS2 domain configuration

3. **CVAT Integration Issues**
   - Ensure function URL is accessible from CVAT
   - Check CVAT logs for error messages
   - Verify model configuration in CVAT admin panel

### Performance Tuning

1. **Confidence Threshold**
   - Start with 0.5 and adjust based on results
   - Lower values will detect more objects but may include false positives

2. **Function Resources**
   - Adjust CPU/memory limits in `function.yaml`
   - Scale replicas based on load requirements

3. **Timeout Settings**
   - Increase timeout in ROS2 bridge if processing is slow
   - Configure CVAT model timeout appropriately

## Advanced Configuration

### Custom Labels

To add custom fruit types or ripeness categories:

1. **Update the converter**
   Edit `cvat_converter.py` to add new fruit type mappings:
   ```python
   self.fruit_type_mapping = {
       'strawberry': 'strawberry',
       'tomato': 'tomato',
       'apple': 'apple',      # Add new types
       'pear': 'pear',
       # ... more types
   }
   ```

2. **Rebuild and redeploy**
   ```bash
   docker build -t aoc-fruit-detector:latest .
   nuctl deploy aoc-fruit-detector --run-image aoc-fruit-detector:latest
   ```

3. **Update CVAT labels**
   Add corresponding labels in CVAT model configuration

### Multi-Model Setup

You can deploy multiple versions of the function for different use cases:

```bash
# Deploy for strawberries only
nuctl deploy aoc-strawberry-detector --env FRUIT_TYPE=strawberry

# Deploy for tomatoes only  
nuctl deploy aoc-tomato-detector --env FRUIT_TYPE=tomato
```