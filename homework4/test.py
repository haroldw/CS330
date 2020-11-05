import os
from google_drive_downloader import GoogleDriveDownloader as gdd

# Need to download the Omniglot dataset -- DON'T MODIFY THIS CELL
if not os.path.isdir('./dream_template'):
    gdd.download_file_from_google_drive(file_id='1O8k6UWSbJOczjQm5-e9g3y1k3FDew1Yn',
                                        dest_path='./dream_template.zip',
                                        unzip=True)
    # !mv dream_template/* ./
    
required_files = ['config.py', 'meta_exploration.py', 'replay.py', 'schedule.py',
                  'policy.py', 'requirements.txt', 'utils.py', 'relabel.py', 
                  'rl.py', 'wrappers.py', 'grid.py', 'render.py', 'city.py']
for f in required_files:
  assert os.path.isfile(f)