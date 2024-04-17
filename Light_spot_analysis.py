import cv2
import numpy as np
import yaml
import json
from influxdb import InfluxDBClient

class ConfigLoader:
    def init(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
            position = config_data.get('position', {})
            std = config_data.get('std', {})
            dispersion = config_data.get('dispersion', {})
            return {'position': position, 'std': std, 'dispersion': dispersion}

    def get_config(self):
        return self.config

class ImageProcessor:
    def init(self, image_path, config_loader):
        self.image_path = image_path
        self.config_loader = config_loader
        self.image = self.load_image()

    def load_image(self):
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path {self.image_path} not found.")
        return image

    def process_image(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        config = self.config_loader.get_config()

        moments = cv2.moments(gray_image)
        x_center = int(moments["m10"] / moments["m00"])
        y_center = int(moments["m01"] / moments["m00"])

        std_dev = np.std(gray_image)
        variance = np.var(gray_image)

        projection_x = np.sum(gray_image, axis=0)
        projection_y = np.sum(gray_image, axis=1)

        np.savetxt("projection_x.txt", projection_x, fmt="%d")
        np.savetxt("projection_y.txt", projection_y, fmt="%d")

        results = {
            "position": {"expected": config['position'], "actual": [x_center, y_center]},
            "std": {"expected": config['std'], "actual": std_dev},
            "dispersion": {"expected": config['dispersion'], "actual": variance}
        }
        return results

class DatabaseManager:
    def __init__(self, host='localhost', port=8086, database='test.db'):
        self.client = InfluxDBClient(host=host, port=port)
        self.client.switch_database(database)

    def send_metrics(self, results):
        json_body = [
            {
                "measurement": "image_analysis",
                "tags": {
                    "user": "example"
                },
                "fields": {
                    "position_x": results["position"]["actual"][0],
                    "position_y": results["position"]["actual"][1],
                    "std_dev": results["std"]["actual"],
                    "variance": results["dispersion"]["actual"]
                }
            }
        ]
        self.client.write_points(json_body)

    def write_to_database(self, measurement, tags, fields):
        json_body = [
            {
                "measurement": measurement,
                "tags": tags,
                "fields": fields
            }
        ]
        self.client.write_points(json_body)

    def read_from_database(self, query):
        result = self.client.query(query)
        return result.raw

class JSONDataSaver:
    def init(self, filename="Output.json"):
        self.filename = filename

    def save_results_to_json(self, results):
        with open(self.filename, 'w') as f:
            json.dump(results, f, indent=4)

class TestManager:
    @staticmethod
    def test_position(expected, actual):
        result = np.isclose(expected, actual, atol=0.01)
        if result:
            print("Position test passed successfully.")
        else:
            print("Position test failed.")
        return result

    @staticmethod
    def test_std(expected, actual):
        result = np.isclose(expected, actual, atol=0.01)
        if result:
            print("Standard deviation test passed successfully.")
        else:
            print("Standard deviation test failed.")
        return result

    @staticmethod
    def test_dispersion(expected, actual):
        result = np.isclose(expected, actual, atol=0.01)
        if result:
            print("Dispersion test passed successfully.")
        else:
            print("Dispersion test failed.")
        return result

if __name__ == "__main__":
    image_path = "image.jpg"
    config_path = "Input.yml"

    config_loader = ConfigLoader(config_path)
    image_processor = ImageProcessor(image_path, config_loader)

    results = image_processor.process_image()

    test_manager = TestManager()
    results['tests'] = {
        'position_test': test_manager.test_position(results['position']['expected'], results['position']['actual']),
        'std_test': test_manager.test_std(results['std']['expected'], results['std']['actual']),
        'dispersion_test': test_manager.test_dispersion(results['dispersion']['expected'], results['dispersion']['actual'])
    }

    json_saver = JSONDataSaver()
    try:
        json_saver.save_results_to_json(results)
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    db_manager = DatabaseManager()
    try:
        db_manager.send_metrics(results)
    except Exception as e:
        print(f"Error sending metrics to database: {e}")