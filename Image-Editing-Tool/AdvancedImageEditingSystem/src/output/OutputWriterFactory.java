package output;


import utils.Constants;

public class OutputWriterFactory {

    public static OutputWriter createWriter(String type, String filePath) throws UnsupportedWriterException {
        // For now, File Writer and Display Writer Are the only output options
        // In the future, we can add more types.
        switch (type.toLowerCase()) {
            case Constants.FILE: return new FileWriter(filePath);
            case Constants.DISPLAY: return new DisplayWriter();
            default: throw new UnsupportedWriterException("Unsupported writer type " + type);
        }
    }
}
