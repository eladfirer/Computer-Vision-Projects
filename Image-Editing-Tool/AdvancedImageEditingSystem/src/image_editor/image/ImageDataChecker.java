package image_editor.image;

import input.exceptions.FormatException;
import utils.Constants;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

// This class lets us add more input readers into our system.
// they can call functions from here to check their data.
// also, it's easier to add new operations and change format by changing only this class.
// Used Ai assistance in order to check the data from the user.
public class ImageDataChecker {
    private static ImageDataChecker instance = null;

    private ImageDataChecker() { }

    public static ImageDataChecker getInstance() {
        if (instance == null) {
            instance = new ImageDataChecker();
        }
        return instance;
    }

    /**
     * Validate that the input path is non-null, points to a readable image file,
     * and can actually be decoded.
     */
    public void checkInput(String path) throws FormatException {
        if (path == null || path.isBlank()) {
            throw new FormatException("'input' must be a non-empty string path");
        }
        File f = new File(path);
        if (!f.isFile()) {
            throw new FormatException("Input file does not exist: " + path);
        }
        try {
            if (ImageIO.read(f) == null) {
                throw new FormatException("File is not a supported image: " + path);
            }
        } catch (IOException e) {
            throw new FormatException("Error reading image file: " + e.getMessage());
        }
    }

    /**
     * @param outputPath  the raw string from JSON (may be empty)
     * @param display     the boolean flag from JSON
     * @throws FormatException if neither display=true nor a valid non-empty outputPath
     *                         or if the output directory doesn’t exist or isn’t writable.
     */
    public void checkOutputAndDisplay(String outputPath, boolean display)
            throws FormatException {
        boolean hasOutput = !outputPath.isBlank();

        if (!display && !hasOutput) {
            throw new FormatException(
                    "Must set either display=true or supply a non-empty output path."
            );
        }

        if (hasOutput) {
            File outFile = new File(outputPath);
            File parent  = outFile.getAbsoluteFile().getParentFile();

            if (parent != null) {
                if (!parent.exists() || !parent.isDirectory()) {
                    throw new FormatException(
                            "Output directory does not exist: " + parent.getPath()
                    );
                }
                if (!parent.canWrite()) {
                    throw new FormatException(
                            "Cannot write to output directory: " + parent.getPath()
                    );
                }
            }
        }
    }
    /**
     * Your existing per‐operation validator.
     */
    public void checkOperation(String type, Map<String,String> params, int index)
        throws FormatException {
        switch (type) {
            case Constants.BRIGHTNESS:
                checkBrightness(type, params, index);
                break;
            case Constants.CONTRAST:
                checkContrast(type, params, index);
                break;
            case Constants.SATURATION:
                checkSaturation(type, params, index);
                break;

            case Constants.SHARPEN:
                checkSharpen(type, params, index);
                break;

            case Constants.BOX:
                checkBox(type, params, index);
                break;

            case Constants.SOBEL:
                checkSobel(type, params, index);
                break;

            default:
                throw new FormatException(
                        "Unsupported operation type '" + type + "' at index " + index + "."
                );
        }
    }

    private void checkContrast(String type, Map<String, String> params, int index) throws FormatException {
        requireKeys(params, index, type, "type", "value");
        checkIsNumber(params.get("value"), index, type, "value");
        double contrastValue = Double.parseDouble(params.get("value"));
        if (Math.abs(contrastValue) > 10000) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has out-of-range 'value': " + contrastValue
            );
        }
    }

    private void checkSharpen(String type, Map<String, String> params, int index) throws FormatException {
        requireKeys(params, index, type, "type", "alpha");
        checkIsNumber(params.get("alpha"), index, type, "alpha");
        double alphaValue = Double.parseDouble(params.get("alpha"));
        if (alphaValue < 0) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has negative 'alpha': " + alphaValue
            );
        }
        if (Math.abs(alphaValue) > 10000) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has out-of-range 'alpha': " + alphaValue
            );
        }
    }

    private void checkSobel(String type, Map<String, String> params, int index) throws FormatException {
        requireKeys(params, index, type, "type", "threshold");
        checkIsNumber(params.get("threshold"), index, type, "threshold");
        double thresholdValue = Double.parseDouble(params.get("threshold"));
        if (thresholdValue < 0) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has negative 'threshold': " + thresholdValue
            );
        }
        if (Math.abs(thresholdValue) > 10000) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has out-of-range 'threshold': " + thresholdValue
            );
        }
    }

    private void checkBrightness(String type, Map<String, String> params, int index) throws FormatException {
        requireKeys(params, index, type, "type", "value");
        checkIsNumber(params.get("value"), index, type, "value");
        double brightnessValue = Double.parseDouble(params.get("value"));
        if (brightnessValue < 0) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has non-positive 'value': " + brightnessValue
            );
        }
        if (Math.abs(brightnessValue) > 10000) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has out-of-range 'value': " + brightnessValue
            );
        }
    }

    private void checkSaturation(String type, Map<String, String> params, int index) throws FormatException {
        requireKeys(params, index, type, "type", "value");
        checkIsNumber(params.get("value"), index, type, "value");
        double saturationValue = Double.parseDouble(params.get("value"));
        if (saturationValue < 0) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has non-positive 'value': " + saturationValue
            );
        }
        if (Math.abs(saturationValue) > 10000) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has out-of-range 'value': " + saturationValue
            );
        }
    }

    public void checkOperations(List<Map<String, String>> ops) throws FormatException {
        // 1) The operations section must appear (i.e. ops != null)
        if (ops == null) {
            throw new FormatException(
                    "The '" + Constants.OPERATIONS + "' section is missing; it must appear."
            );
        }

        // 2) Validate each operation
        for (int i = 0; i < ops.size(); i++) {
            Map<String,String> params = ops.get(i);

            if (params.size() == 1 && params.containsKey("") && "".equals(params.get(""))) {
                throw new FormatException(
                        "Operation at index " + i + " is not a valid operation."
                );
            }

            String type = params.get(Constants.TYPE);
            if (type == null) {
                throw new FormatException(
                        "Operation at index " + i + " is missing the '"
                                + Constants.TYPE + "' field."
                );
            }

            checkOperation(type.toLowerCase(), params, i);
        }
    }



    private void checkBox(String type, Map<String, String> params, int index) throws FormatException {
        requireKeys(params, index, type, "type", "width", "height");
        checkIsInteger(params.get("width"), index, type, "width");
        checkIsInteger(params.get("height"), index, type, "height");
        int w = Integer.parseInt(params.get("width"));
        if (w < 1) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has non-positive 'width': " + w
            );
        }
        if (w > 1000) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has too large 'width': " + w
            );
        }
        int h = Integer.parseInt(params.get("height"));
        if (h < 1) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has non-positive 'height': " + h
            );
        }
        if (h > 1000) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index
                            + " has too large 'height': " + h
            );
        }
    }

    private void requireKeys(
            Map<String,String> params,
            int index,
            String type,
            String... required
    ) throws FormatException {
        // ensure required keys are present
        for (String key : required) {
            if (!params.containsKey(key)) {
                throw new FormatException(
                        "Operation '" + type + "' at index " + index +
                                " is missing required key '" + key + "'."
                );
            }
        }

        // ensure no extra keys
        for (String key : params.keySet()) {
            boolean ok = false;
            for (String r : required) if (r.equals(key)) { ok = true; break; }
            if (!ok) {
                throw new FormatException(
                        "Operation '" + type + "' at index " + index +
                                " has unexpected key '" + key + "'."
                );
            }
        }
    }

    private void checkIsNumber(
            String raw,
            int index,
            String type,
            String param
    ) throws FormatException {
        try {
            Double.parseDouble(raw);
        } catch (Exception e) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index +
                            " has non-numeric '" + param + "': " + raw
            );
        }
    }

    private void checkIsInteger(
            String raw,
            int index,
            String type,
            String param
    ) throws FormatException {
        try {
            Integer.parseInt(raw);
        } catch (Exception e) {
            throw new FormatException(
                    "Operation '" + type + "' at index " + index +
                            " has non-integer '" + param + "': " + raw
            );
        }
    }
}
