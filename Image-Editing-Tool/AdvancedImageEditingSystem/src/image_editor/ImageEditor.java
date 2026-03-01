package image_editor;

import image_editor.image.DataContainer;
import operations.Operation;
import input.InputReader;
import input.InputReaderFactory;
import output.OutputWriter;
import program.Program;
import utils.Constants;


import java.io.IOException;


public class ImageEditor implements Program {
    private final String filePath;
    private final String inputType;


    public ImageEditor(String filePath, String inputType) {
        this.filePath = filePath;
        this.inputType = inputType;
    }


    @Override
    public void execute() throws IOException {
        // the input reader factory gets the program type and the reader type
        InputReader reader = InputReaderFactory.createReader(inputType, filePath, Constants.IMAGE_EDITOR);

        DataContainer dataContainer = reader.read();
        runOperations(dataContainer);
        outputImage(dataContainer);
    }


    private void runOperations(DataContainer dataContainer) {
        for (Operation op : dataContainer.getOperations()) {
            op.activateOnImage(dataContainer.getImage());
        }
    }
    private void outputImage(DataContainer dataContainer) {
        for (OutputWriter output : dataContainer.getOutputs()){
            output.outputImage(dataContainer.getImage());
        }
    }
}
