import os
from dataclasses import dataclass, field
from typing import List

from fid import calculate_fid, load_images


@dataclass
class MethodsEvaluation:
    name: str
    batch_size: int

    def __post_init__(self):
        self.generated_path = f'generated/{self.name}'
        self.real_path = f'original/{self.name}'
        
        self.generated_images = load_images(self.generated_path)
        self.real_images = load_images(self.real_path)


    def get_the_fid_score(self) -> float:
        fid_value = calculate_fid(images1=self.generated_images, 
                                  images2=self.real_images, 
                                  use_multiprocessing=False, 
                                  batch_size=self.batch_size
                                  )
        return fid_value

if __name__ == "__main__":
    evaluation_instance = MethodsEvaluation(name="Pix2Pix", batch_size=64)
    print("FID score of method is ", evaluation_instance.get_the_fid_score())