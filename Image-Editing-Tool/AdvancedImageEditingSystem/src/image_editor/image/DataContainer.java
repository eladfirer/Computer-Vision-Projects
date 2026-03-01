package image_editor.image;

import operations.Operation;
import output.OutputWriter;

import java.util.List;

public class DataContainer {
    private Image image;
    private List<Operation> operations;
    private List<OutputWriter> outputs;

    public DataContainer(Image image, List<Operation> operations, List<OutputWriter> outputs) {
        this.image = image;
        this.operations = operations;
        this.outputs = outputs;
    }

    public Image getImage() {
        return image;
    }

    public List<Operation> getOperations() {
        return operations;
    }

    public List<OutputWriter> getOutputs() {
        return outputs;
    }
}
