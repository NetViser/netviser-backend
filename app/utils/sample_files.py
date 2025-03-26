# Description: This file contains the mapping of sample files to their respective bucket keys and featured attacks.

sample_file_bucket_key_mapping = {
    "ssh-ftp.csv": "sample/model-applied/ssh-ftp.csv",
    "ddos-ftp.csv": "sample/model-applied/ddos-ftp.csv",
    "ftp_patator_occurence.csv": "sample/model-applied/ftp_patator_occurence.csv",
    "portscan_dos_hulk_slowloris.csv": "sample/model-applied/portscan_dos_hulk_slowloris.csv",
    "portscan_dos_hulk.csv": "sample/model-applied/portscan_dos_hulk.csv",
    "portscan.csv": "sample/model-applied/portscan.csv",
    "ftp": "sample/model-applied/ftp.csv",
    "ddos-large.csv": "sample/model-applied/ddos-large.csv",
    "dos-hulk.csv": "sample/model-applied/dos-hulk.csv",
}

sample_file_featured_attacks_mapping = {
    "ddos-ftp.csv": ["DDoS", "FTP-Patator"],
    "ssh-ftp.csv": ["SSH-Patator", "FTP-Patator"],
    "ftp_patator_occurence.csv": ["FTP-Patator"],
    "portscan_dos_hulk_slowloris.csv": ["DoS Hulk", "DoS Slowloris", "PortScan"],
    "portscan_dos_hulk.csv": ["DoS Hulk", "PortScan"],
    "portscan.csv": ["PortScan"],
    "ftp.csv": ["FTP-Patator"],
    "ddos-large.csv": ["DDoS"],
    "dos-hulk.csv": ["DoS Hulk"],
}