#!/bin/bash
directory=$1
archive_name=$(basename ${directory}).tar.gz

tar -zcvf ${archive_name} -C ${directory} .
