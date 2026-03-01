package input;

import input.exceptions.UnsupportedInputException;
import input.json_reader.image.ImageJsonDataExtractor;
import input.json_reader.image.ImageJsonFormatChecker;
import input.json_reader.JsonReader;
import utils.Constants;

public class InputReaderFactory {
    public static InputReader createReader(String type, String fileName, String programType) throws UnsupportedInputException {
        switch (type.toLowerCase()) {
            case Constants.JSON: {
                return getJsonReader(fileName, programType);
            }
            default: throw new UnsupportedInputException("Unsupported input type " + type);
        }
    }




    private static JsonReader getJsonReader(String fileName, String programType) throws UnsupportedInputException {
        switch (programType.toLowerCase()) {
            case Constants.IMAGE_EDITOR:
                return new JsonReader(
                        fileName,
                        new ImageJsonFormatChecker(),
                        new ImageJsonDataExtractor()
                );
            default:
                throw new UnsupportedInputException("Unsupported program type " + programType);
        }
    }

}
