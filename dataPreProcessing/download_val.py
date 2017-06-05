# The files are downloaded to the same directory this script in located. You can use 
# this script several times to resume downloading the data. Therefore, those files 
# downloaded correctly, will not be downloaded again. 


import urllib2
import os

class downloader(object):

    def __init__(self, urls):
        for url in urls:
            self.dl(url)

    def dl(self, url):
        file_name = url.split('/')[-1]
        u = urllib2.urlopen(url)
        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        print("Downloading: %s Bytes: %s" % (file_name, file_size))

        file_size_dl = 0
        block_sz = 65536
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8) * (len(status) + 1)
            print(status)

        f.close()


if __name__ == "__main__":
    urls=["http://158.109.8.102/FirstImpressionsV2/val-1.zip",
          "http://158.109.8.102/FirstImpressionsV2/val-2.zip",
          "http://158.109.8.102/FirstImpressionsV2/val-transcription.zip"]

    new_urls = []

    for url in urls:
        file_name = url.split('/')[-1]
        u = urllib2.urlopen(url)
        meta = u.info()
        remote_file_size = int(meta.getheaders("Content-Length")[0])

        if os.path.isfile(file_name):
            statinfo = os.stat(file_name)
            if statinfo.st_size==remote_file_size:
                print("File %s is available. Checking next file." % (file_name))
                continue
        new_urls.append(url)
    downloader(new_urls)

