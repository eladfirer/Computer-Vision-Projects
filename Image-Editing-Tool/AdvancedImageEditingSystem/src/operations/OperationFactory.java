package operations;

import utils.Constants;
import java.util.Map;

public class OperationFactory {

    public static Operation createOperation(String type, Map<String,String> params) {
        switch (type) {
            case Constants.BRIGHTNESS:
                double b = Double.parseDouble(params.get("value"));
                return new BrightnessOperation(b);

            case Constants.CONTRAST:
                double c = Double.parseDouble(params.get("value"));
                return new ContrastOperation(c);

            case Constants.SATURATION:
                double s = Double.parseDouble(params.get("value"));
                return new SaturationOperation(s);

            case Constants.SHARPEN:
                double a = Double.parseDouble(params.get("alpha"));
                return new SharpenOperation(a);

            case Constants.BOX:
                int w = Integer.parseInt(params.get("width"));
                int h = Integer.parseInt(params.get("height"));
                return new BoxBlurOperation(w, h);

            case Constants.SOBEL:
                double t = Double.parseDouble(params.get("threshold"));
                return new SobelOperation(t);

            default:
                throw new IllegalArgumentException("Unsupported operation type: " + type);
        }
    }
}

