"""This script downloads all datasets used in the book."""
from dcss import download_dataset


# VDEM
vdem_data_url = "https://www.dropbox.com/scl/fo/6ay4x2qo4svyo92wbvlxt/ACtUxCDoLYxLujkekHdXiJ4?rlkey=lhmhiasjkv3ndvyxjxapi24sk&st=2p76a0dw&dl=0"
download_dataset(vdem_data_url, save_path='vdem/')

# Freedom House
fh_url = 'https://www.dropbox.com/scl/fo/fnw5yrslxza9plhqnqhxr/AA7997oGIdd3k3EjluHyLBc?rlkey=hr93qtcdp6uh7d3lsfzbc6nr6&st=bz0xzw41&dl=0'
download_dataset(fh_url, 'freedom_house/')

# VDEM + Freedom House
vif_combo_link = "https://www.dropbox.com/scl/fo/q66z60y6r3ql8tnl9stp7/APPon9yB-tJEm_zq-JCdqcA?rlkey=hrac95v1zcdy3wur86dujtpxw&st=5v1od1f1&dl=0"
download_dataset(vif_combo_link, 'vdem_internet_freedom_combined/')

# Russian Troll Tweets
russian_troll_data_url = "https://www.dropbox.com/scl/fo/a3uxioa2wd7k8x8nas0iy/AH5qjXAZvtFpZeIID0sZ1xA?rlkey=p1471igxmzgyu3lg2x93b3r1y&st=xvhtn8gi&dl=0"
download_dataset(russian_troll_data_url, 'russian_troll_tweets/')

# Copenhagen Networks Study (CNS)
cns_url = "https://www.dropbox.com/scl/fo/svfktjetp2j8we2mh2s3g/ADogVe6DhJPEwIEsiGJndjI?rlkey=fozovcld3xdztawxj2hxzvxzv&st=016vi6lr&dl=0"
download_dataset(cns_url, 'copenhagen_networks_study/')

# US Election 2020
us_election_2020_data_url = "https://www.dropbox.com/scl/fo/gcotab57xtv9a0ga5vums/ANB2gm71cIXW1NcLwA5ezXY?rlkey=nai1uun6mkl10a66ekzs692ux&st=rmbsufjx&dl=0"
download_dataset(us_election_2020_data_url, '2020_election/')

# SocioPatterns
sp_url = "https://www.dropbox.com/scl/fo/wbj6l2tyoc67o3vlonxp5/AERxFPRIhfgWG_MaVK_rjM4?rlkey=48x2taz2t5mru1j2ucjf670a8&st=e5qs5gw8&dl=0"
download_dataset(sp_url, save_path="SocioPatterns/")

# Canadian Hansard
ca_hansard_data_url = "https://www.dropbox.com/scl/fo/5voxfrx6qeqgdrjuc979k/AD63UZhKpxF64b58Jp65w18?rlkey=2bbaqw1bwjgvodbwqhox494e2&st=99fldjgr&dl=0"
download_dataset(ca_hansard_data_url, 'canadian_hansard/')

# British Hansard (2016-2020)
years = [2016, 2017, 2018, 2019, 2020]
columns = [
    'speech',
    'speakername',
    'party',
    'constituency',
    'year'
]
dfs = []

for year in years:
    bh_year_url=f'https://www.dropbox.com/scl/fi/c9d1aqzrage1juf276nvd/british_hansard_{year}.csv?rlkey=ilyn06y4hw4jocr4fhq6w6olk&st=ioslogzq&dl=1'
    download_dataset(bh_year_url, save_path=f'british_hansard/bh{year}.csv')
