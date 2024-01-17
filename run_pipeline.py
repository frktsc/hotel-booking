from pipelines.training_pipeline import Train_Pipeline
from  zenml.client import Client



if __name__=="__main__":
    print('mlf_flowuurl',Client().active_stack.experiment_tracker.get_tracking_uri())
    Train_Pipeline(data_path=r'C:\Users\furkanbaba\Desktop\furkanbaba\coding\hotel-booking\data\data_hotel_booking.csv')    
# mlflow ui --backend-store-uri "file:C:\Users\furkanbaba\AppData\Roaming\zenml\local_stores\89d0ad23-619c-43ce-9b97-ce7d8b34b0b4\mlruns"