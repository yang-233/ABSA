from IPython import display
import matplotlib.pyplot as plt
def _draw(figsize:Tuple[int, int], epoches:List[int],
          train_loss:List[float], train_acc:List[float], train_f1:List[float],
          test_loss:List[float], test_acc:List[float], test_f1:List[float], only_test:bool=False):

    plt.figure(12, figsize=figsize)
        
    plt.subplot(121)
    plt.plot(epoches, train_loss, c="red", label="Train loss")
    plt.plot(epoches, test_loss, c="blue", label="Test loss")
    plt.xlabel("Epoches")
    plt.title("Loss")
    
    if not only_test:
        plt.plot(epoches, train_acc, c="red", label="Train accuracy")
        plt.plot(epoches, train_f1, c="purple", label="Train f1")
    
    plt.plot(epoches, test_acc, c="green", label="Test accuracy")
    plt.plot(epoches, test_f1, c="blue", label="Test f1")
    plt.xlabel("Epoches")
    plt.title("Accuracy and F1")
    plt.subplot(122)
    
    display.display(plt.gcf())
    display.clear_output(wait=True)
