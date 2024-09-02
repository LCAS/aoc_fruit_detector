"""
Started by: Usman Zahidi (uz) {20/08/24}
"""
# general imports
import os,logging,traceback
import shutil, requests,zipfile

# UZ:utils for download to serve both trainer and predictor

class LearnerUtils:

    def __init__(self, config_data):
        #UZ:load config data into object variables
        #UZ: Files
        self.model_file                     = config_data['files']['model_file']
        self.train_annotation_file          = config_data['files']['train_annotation_file']
        self.test_annotation_file           = config_data['files']['test_annotation_file']
        self.test_metadata_catalog_file     = config_data['files']['test_metadata_catalog_file']
        self.model_url                      = config_data['files']['model_url']
        self.meta_catalog_url               = config_data['files']['meta_catalog_url']
        self.train_catalog_url              = config_data['files']['train_catalog_url']
        self.train_dataset_catalog_file     = config_data['files']['train_dataset_catalog_file']
        # UZ: Datasets
        self.train_dataset_name             = config_data['datasets']['train_dataset_name']
        self.test_dataset_name              = config_data['datasets']['test_dataset_name']
        self.dataset_train_annotation_url   = config_data['datasets']['dataset_train_annotation_url']
        self.dataset_train_images_url       = config_data['datasets']['dataset_train_images_url']
        self.dataset_test_annotation_url    = config_data['datasets']['dataset_test_annotation_url']
        self.dataset_test_images_url        = config_data['datasets']['dataset_test_images_url']
        # UZ: Directories
        self.train_image_dir                = config_data['directories']['train_image_dir']
        self.test_image_dir                 = config_data['directories']['test_image_dir']

    def call_download(self)->None:
        if not os.path.exists(self.model_file):
            model_dir, local_filename = os.path.split(self.model_file)
            self._download(self.model_url, model_dir, local_filename)

        self._download_datasets(self.train_annotation_file, self.dataset_train_annotation_url,
                                self.train_dataset_name,self.train_image_dir,self.dataset_train_images_url)
        self._download_datasets(self.test_annotation_file, self.dataset_test_annotation_url,
                                self.test_dataset_name,self.test_image_dir,self.dataset_test_images_url)

        self._download_catalogs(self.test_metadata_catalog_file, self.meta_catalog_url, self.train_dataset_catalog_file,
                           self.train_catalog_url)


    def _download_catalogs(self, test_metadata_catalog_file, meta_catalog_url, train_dataset_catalog_file,
                          train_catalog_url):
        if not os.path.exists(test_metadata_catalog_file):
            #UZ: download metadata catalog
            catalog_dir, local_filename = os.path.split(test_metadata_catalog_file)
            self._download(meta_catalog_url, catalog_dir, local_filename)
            #UZ: download train catalog
            catalog_dir, local_filename = os.path.split(train_dataset_catalog_file)
            self._download(train_catalog_url, catalog_dir, local_filename)

    def _download_datasets(self,train_annotation_file, dataset_train_annotation_url,
                           train_dataset_name,train_image_dir,dataset_images_url):
        if not os.path.exists(train_annotation_file):
            model_dir, local_filename = os.path.split(train_annotation_file)
            self._download(dataset_train_annotation_url, model_dir, local_filename)
            local_filename=train_dataset_name + '.zip'
            self._download(dataset_images_url, train_image_dir, local_filename)
            path_to_zip_file=os.path.join(train_image_dir, local_filename)
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(train_image_dir)


    def _download(self, url: str, folder_name: str, local_filename: str)->None:
        try:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)  # create folder if it does not exist

            file_path = os.path.join("{}/{}".format(folder_name, local_filename))

            with requests.get(url, stream=True) as r:
                with open(file_path, 'wb') as f:
                    print("saving to", os.path.abspath(file_path))
                    shutil.copyfileobj(r.raw, f)
        except Exception as e:
            logging.error(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)

