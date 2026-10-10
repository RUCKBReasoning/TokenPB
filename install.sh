#!/usr/bin/env bash

# error codes
# 0 - exited without problems
# 1 - parameters not supported were used or some unexpected error occurred
# 2 - OS not supported by this script
# 3 - installed version of ossutil is up to date
# 4 - supported unzip tools are not availabale

set -e

#when adding a tool to the list make sure to also add its corresponding command further in the script
unzip_tools_list=('unzip' '7z' 'busybox')

usage() { echo "Usage: sudo -v ; curl https://gosspublic.alicdn.com/ossutil/install.sh | sudo bash" 1>&2; exit 1; }


#create tmp directory and move to it with macOS compatibility fallback
tmp_dir=$(mktemp -d 2>/dev/null || mktemp -d -t 'ossutil-install.XXXXXXXXXX')
cd "$tmp_dir"


#make sure unzip tool is available and choose one to work with
set +e
for tool in ${unzip_tools_list[*]}; do
    trash=$(hash "$tool" 2>>errors)
    if [ "$?" -eq 0 ]; then
        unzip_tool="$tool"
        break
    fi
done  
set -e

# exit if no unzip tools available
if [ -z "$unzip_tool" ]; then
    printf "\nNone of the supported tools for extracting zip archives (${unzip_tools_list[*]}) were found. "
    printf "Please install one of them and try again.\n\n"
    exit 4
fi

#check installed version of ossutil to determine if update is necessary
version=$(ossutil --version 2>>errors | head -n 1)
current_version=$(curl -fsS https://gosspublic.alicdn.com/ossutil/version.txt)

if [ "$version" = "$current_version" ]; then
    printf "\nThe latest version of ossutil ${version} is already installed.\n\n"
    exit 3
fi


#detect the platform
cmd_suffix=""
OS="$(uname)"
case $OS in
  Linux)
    OS='linux'
    ;;
  Darwin)
    OS='mac'
    binTgtDir=/usr/local/bin
    cmd_suffix='mac'
    ;;
  *)
    echo 'OS not supported'
    exit 2
    ;;
esac

OS_type="$(uname -m)"
case "$OS_type" in
  x86_64|amd64)
    OS_type='amd64'
    cmd_suffix=${cmd_suffix}'64'
    ;;
  i?86|x86)
    OS_type='386'
    cmd_suffix=${cmd_suffix}'32'
    ;;
  aarch64|arm64)
    OS_type='arm64'
    cmd_suffix=${cmd_suffix}'64'
    ;;
  arm*)
    OS_type='arm'
    cmd_suffix=${cmd_suffix}'32'
    ;;
  *)
    echo 'OS type not supported'
    exit 2
    ;;
esac


#download and unzip
download_link="https://gosspublic.alicdn.com/ossutil/current/ossutil-${OS}-${OS_type}.zip"
ossutil_zip="ossutil-${OS}-${OS_type}.zip"

curl -OfsS "$download_link"
unzip_dir="tmp_unzip_dir_for_ossutil"
# there should be an entry in this switch for each element of unzip_tools_list
case "$unzip_tool" in
  'unzip')
    unzip -a "$ossutil_zip" -d "$unzip_dir"
    ;;
  '7z')
    7z x "$ossutil_zip" "-o$unzip_dir"
    ;;
  'busybox')
    mkdir -p "$unzip_dir"
    busybox unzip "$ossutil_zip" -d "$unzip_dir"
    ;;
esac

cd $unzip_dir/*

#mounting ossutil to environment

case "$OS" in
  'linux')
    #binary
    cp ossutil /usr/bin/ossutil.new
    chmod 755 /usr/bin/ossutil.new
    chown root:root /usr/bin/ossutil.new
    mv /usr/bin/ossutil.new /usr/bin/ossutil
    ln -sf /usr/bin/ossutil /usr/bin/ossutil${cmd_suffix}
    ;;
  'mac')
    #binary
    mkdir -m 0555 -p ${binTgtDir}
    cp ossutil ${binTgtDir}/ossutil.new
    cp ossutil${cmd_suffix} ${binTgtDir}/ossutil${cmd_suffix}.new
    mv ${binTgtDir}/ossutil.new ${binTgtDir}/ossutil
    mv ${binTgtDir}/ossutil${cmd_suffix}.new ${binTgtDir}/ossutil${cmd_suffix}
    chmod a=x ${binTgtDir}/ossutil
    chmod a=x ${binTgtDir}/ossutil${cmd_suffix}
    ;;
  *)
    echo 'OS not supported'
    exit 2
esac


#update version variable post install
version=$(ossutil --version 2>>errors | head -n 1)

printf "\n${version} has successfully installed."
printf '\nNow run "ossutil config" for setup. Check https://help.aliyun.com/document_detail/50452.html for more details.\n\n'
exit 0
