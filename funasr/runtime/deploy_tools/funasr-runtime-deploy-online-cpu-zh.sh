#!/usr/bin/env bash

scriptVersion="0.0.6"
scriptDate="20230705"


# Set color
RED="\033[31;1m"
GREEN="\033[32;1m"
YELLOW="\033[33;1m"
BLUE="\033[34;1m"
CYAN="\033[36;1m"
PLAIN="\033[0m"

# Info messages
DONE="${GREEN}[DONE]${PLAIN}"
ERROR="${RED}[ERROR]${PLAIN}"
WARNING="${YELLOW}[WARNING]${PLAIN}"

# Font Format
BOLD="\033[1m"
UNDERLINE="\033[4m"

# Current folder
CUR_DIR=`pwd`
SUDO_CMD="sudo"


DEFAULT_DOCKER_OFFLINE_CPU_ZH_LISTS_OSS="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/docker_lists/docker_online_cpu_zh_lists"
DEFAULT_DOCKER_OFFLINE_CPU_ZH_LISTS_GIT="https://raw.githubusercontent.com/alibaba-damo-academy/FunASR/main/funasr/runtime/docs/docker_online_cpu_zh_lists"
DEFAULT_DOCKER_IMAGE_LISTS=$DEFAULT_DOCKER_OFFLINE_CPU_ZH_LISTS_OSS
DEFAULT_FUNASR_DOCKER_URL="registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr"
DEFAULT_FUNASR_RUNTIME_RESOURCES="funasr-runtime-resources"
DEFAULT_FUNASR_LOCAL_WORKSPACE=${CUR_DIR}/${DEFAULT_FUNASR_RUNTIME_RESOURCES}
DEFAULT_FUNASR_CONFIG_DIR=""
DEFAULT_FUNASR_CONFIG_DIR_BAK="/var/funasr"
DEFAULT_FUNASR_CONFIG_FILE="${DEFAULT_FUNASR_CONFIG_DIR}/config"
DEFAULT_FUNASR_SERVER_CONFIG_FILE="${DEFAULT_FUNASR_CONFIG_DIR}/server_config"
DEFAULT_FUNASR_PROGRESS_TXT="${DEFAULT_FUNASR_CONFIG_DIR}/progress.txt"
DEFAULT_FUNASR_SERVER_LOG="${DEFAULT_FUNASR_CONFIG_DIR}/server_console.log"
DEFAULT_FUNASR_WORKSPACE_DIR="/workspace/models"
DEFAULT_DOCKER_PORT="10095"
DEFAULT_PROGRESS_FILENAME="progress.txt"
DEFAULT_SERVER_EXEC_NAME="funasr-wss-server"
DEFAULT_DOCKER_EXEC_DIR="/workspace/FunASR/funasr/runtime/websocket/build/bin"
DEFAULT_DOCKER_EXEC_PATH=${DEFAULT_DOCKER_EXEC_DIR}/${DEFAULT_SERVER_EXEC_NAME}
DEFAULT_SAMPLES_NAME="funasr_samples"
DEFAULT_SAMPLES_DIR="samples"
DEFAULT_SAMPLES_URL="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/${DEFAULT_SAMPLES_NAME}.tar.gz"

SAMPLE_CLIENTS=( \
"Python" \
"Linux_Cpp" \
)
DOCKER_IMAGES=()

# Handles the download progress bar
asr_percent_int=0
vad_percent_int=0
punc_percent_int=0
asr_title="Downloading"
asr_percent="0"
asr_speed="0KB/s"
asr_revision=""
vad_title="Downloading"
vad_percent="0"
vad_speed="0KB/s"
vad_revision=""
punc_title="Downloading"
punc_percent="0"
punc_speed="0KB/s"
punc_revision=""
serverProgress(){
    status_flag="STATUS:"
    stage=0
    wait=0
    server_status=""

    while true
    do
        if [ -f "$DEFAULT_FUNASR_PROGRESS_TXT" ]; then
            break
        else
            sleep 1
            let wait=wait+1
            if [ ${wait} -ge 6 ]; then
                break
            fi
        fi
    done

    if [ ! -f "$DEFAULT_FUNASR_PROGRESS_TXT" ]; then
        echo -e "    ${RED}The note of progress does not exist.($DEFAULT_FUNASR_PROGRESS_TXT) ${PLAIN}"
        return 98
    fi

    stage=1
    while read line
    do
        if [ $stage -eq 1 ]; then
            result=$(echo $line | grep "STATUS:")
            if [ "$result" != "" ]; then
                stage=2
                server_status=${line#*:}
                status=`expr $server_status + 0`
                if [ $status -eq 99 ]; then
                    stage=99
                fi
                continue
            fi
        elif [ $stage -eq 2 ]; then
            result=$(echo $line | grep "ASR")
            if [ "$result" != "" ]; then
                stage=3
                continue
            fi
        elif [ $stage -eq 3 ]; then
            result=$(echo $line | grep "VAD")
            if [ "$result" != "" ]; then
                stage=4
                continue
            fi
            result=$(echo $line | grep "title:")
            if [ "$result" != "" ]; then
                asr_title=${line#*:}
                continue
            fi
            result=$(echo $line | grep "percent:")
            if [ "$result" != "" ]; then
                asr_percent=${line#*:}
                continue
            fi
            result=$(echo $line | grep "speed:")
            if [ "$result" != "" ]; then
                asr_speed=${line#*:}
                continue
            fi
            result=$(echo $line | grep "revision:")
            if [ "$result" != "" ]; then
                asr_revision=${line#*:}
                continue
            fi
        elif [ $stage -eq 4 ]; then
            result=$(echo $line | grep "PUNC")
            if [ "$result" != "" ]; then
                stage=5
                continue
            fi
            result=$(echo $line | grep "title:")
            if [ "$result" != "" ]; then
                vad_title=${line#*:}
                continue
            fi
            result=$(echo $line | grep "percent:")
            if [ "$result" != "" ]; then
                vad_percent=${line#*:}
                continue
            fi
            result=$(echo $line | grep "speed:")
            if [ "$result" != "" ]; then
                vad_speed=${line#*:}
                continue
            fi
            result=$(echo $line | grep "revision:")
            if [ "$result" != "" ]; then
                vad_revision=${line#*:}
                continue
            fi
        elif [ $stage -eq 5 ]; then
            result=$(echo $line | grep "DONE")
            if [ "$result" != "" ]; then
                # Done and break.
                stage=6
                break
            fi
            result=$(echo $line | grep "title:")
            if [ "$result" != "" ]; then
                punc_title=${line#*:}
                continue
            fi
            result=$(echo $line | grep "percent:")
            if [ "$result" != "" ]; then
                punc_percent=${line#*:}
                continue
            fi
            result=$(echo $line | grep "speed:")
            if [ "$result" != "" ]; then
                punc_speed=${line#*:}
                continue
            fi
            result=$(echo $line | grep "revision:")
            if [ "$result" != "" ]; then
                punc_revision=${line#*:}
                continue
            fi
        elif [ $stage -eq 99 ]; then
            echo -e "    ${RED}ERROR: $line${PLAIN}"
        fi
    done < $DEFAULT_FUNASR_PROGRESS_TXT

    if [ $stage -ne 99 ]; then
        drawProgress "ASR " $asr_title $asr_percent $asr_speed $asr_revision $asr_percent_int
        asr_percent_int=$?
        drawProgress "VAD " $vad_title $vad_percent $vad_speed $vad_revision $vad_percent_int
        vad_percent_int=$?
        drawProgress "PUNC" $punc_title $punc_percent $punc_speed $punc_revision $punc_percent_int
        punc_percent_int=$?
    fi

    return $stage
}

drawProgress(){
    model=$1
    title=$2
    percent_str=$3
    speed=$4
    revision=$5
    latest_percent=$6

    progress=0
    if [ ! -z "$percent_str" ]; then
        progress=`expr $percent_str + 0`
        latest_percent=`expr $latest_percent + 0`
        if [ $progress -ne 0 ] && [ $progress -lt $latest_percent ]; then
            progress=$latest_percent
        fi
    fi

    loading_flag="Loading"
    if [ "$title" = "$loading_flag" ]; then
        progress=100
    fi

    i=0
    str=""
    let max=progress/2
    while [ $i -lt $max ]
    do
        let i++
        str+='='
    done
    let color=36
    let index=max*2
    if [ -z "$speed" ]; then
        printf "\r    \e[0;${CYAN}[%s][%-11s][%-50s][%d%%][%s]\e[0m" "$model" "$title" "$str" "$$index" "$revision"
    else
        printf "\r    \e[0;${CYAN}[%s][%-11s][%-50s][%3d%%][%8s][%s]\e[0m" "$model" "$title" "$str" "$index" "$speed" "$revision"
    fi
    printf "\n"

    return $progress
}

menuSelection(){
    local menu
    menu=($(echo "$@"))
    result=1
    show_no=1
    menu_no=0
    len=${#menu[@]}

    while true
    do
        echo -e "    ${BOLD}${show_no})${PLAIN} ${menu[menu_no]}"

        let show_no++
        let menu_no++
        if [ $menu_no -ge $len ]; then
            break
        fi
    done

    while true
    do
        echo -e "  Enter your choice, default(${CYAN}1${PLAIN}): \c"
        read result
        if [ -z "$result" ]; then
            result=1
        fi

        expr $result + 0 &>/dev/null
        if [ $? -eq 0 ]; then
            if [ $result -ge 1 ] && [ $result -le $len ]; then
                break
            else
                echo -e "    ${RED}Input error, please input correct number!${PLAIN}"
            fi
        else
            echo -e "    ${RED}Input error, please input correct number!${PLAIN}"
        fi
    done
    
    return $result
}


full_path=""
relativePathToFullPath(){
    relativePath=$1
    firstChar=${relativePath: 0: 1}
    if [[ "$firstChar" == "" ]]; then
        full_path=$relativePath
    elif [[ "$firstChar" == "/" ]]; then
        full_path=$relativePath
    fi

    tmpPath1=`dirname $relativePath`
    tmpFullpath1=`cd $tmpPath1 && pwd`
    tmpPath2=`basename $relativePath`
    full_path=${tmpFullpath1}/${tmpPath2}
}

initConfiguration(){
    if [ -z "$DEFAULT_FUNASR_CONFIG_DIR" ];then
        DEFAULT_FUNASR_CONFIG_DIR="$HOME"
        if [ -z "$DEFAULT_FUNASR_CONFIG_DIR" ];then
            $DEFAULT_FUNASR_CONFIG_DIR=$(echo ~/)
            if [ -z "$DEFAULT_FUNASR_CONFIG_DIR" ];then
                $DEFAULT_FUNASR_CONFIG_DIR=$DEFAULT_FUNASR_CONFIG_DIR_BAK
            fi
        fi
        DEFAULT_FUNASR_CONFIG_DIR=${DEFAULT_FUNASR_CONFIG_DIR}/.funasr
    fi

    if [ ! -z "$DEFAULT_FUNASR_CONFIG_DIR" ]; then
        $SUDO_CMD mkdir -p $DEFAULT_FUNASR_CONFIG_DIR
    else
        echo -e "    ${RED}DEFAULT_FUNASR_CONFIG_DIR is empty!${PLAIN}"
        exit 1
    fi
    if [ ! -d "$DEFAULT_FUNASR_CONFIG_DIR" ]; then
        echo -e "    ${RED}${DEFAULT_FUNASR_CONFIG_DIR} does not exist!${PLAIN}"
        exit 2
    fi

    DEFAULT_FUNASR_CONFIG_FILE="${DEFAULT_FUNASR_CONFIG_DIR}/config"
    DEFAULT_FUNASR_SERVER_CONFIG_FILE="${DEFAULT_FUNASR_CONFIG_DIR}/server_config"
    DEFAULT_FUNASR_PROGRESS_TXT="${DEFAULT_FUNASR_CONFIG_DIR}/progress.txt"
    DEFAULT_FUNASR_SERVER_LOG="${DEFAULT_FUNASR_CONFIG_DIR}/server_console.log"

    if [ ! -f $DEFAULT_FUNASR_CONFIG_FILE ]; then
        $SUDO_CMD touch $DEFAULT_FUNASR_CONFIG_FILE
    fi
    if [ ! -f $DEFAULT_FUNASR_SERVER_CONFIG_FILE ]; then
        $SUDO_CMD touch $DEFAULT_FUNASR_SERVER_CONFIG_FILE
    fi
}

initParameters(){
    # Init workspace in local by new parameters.
    PARAMS_FUNASR_SAMPLES_LOCAL_PATH=${PARAMS_FUNASR_LOCAL_WORKSPACE}/${DEFAULT_SAMPLES_NAME}.tar.gz
    PARAMS_FUNASR_SAMPLES_LOCAL_DIR=${PARAMS_FUNASR_LOCAL_WORKSPACE}/${DEFAULT_SAMPLES_DIR}
    PARAMS_FUNASR_LOCAL_MODELS_DIR="${PARAMS_FUNASR_LOCAL_WORKSPACE}/models"

    if [ ! -z "$PARAMS_FUNASR_LOCAL_WORKSPACE" ]; then
        mkdir -p $PARAMS_FUNASR_LOCAL_WORKSPACE
    fi
    if [ ! -z "$PARAMS_FUNASR_LOCAL_MODELS_DIR" ]; then
        mkdir -p $PARAMS_FUNASR_LOCAL_MODELS_DIR
    fi
}

# Parse the parameters from the docker list file.
docker_info_cur_key=""
docker_info_cur_val=""
findTypeOfDockerInfo(){
    line=$1
    result=$(echo $line | grep ":")
    if [ "$result" != "" ]; then
        docker_info_cur_key=$result
        docker_info_cur_val=""
    else
        docker_info_cur_val=$(echo $line)
    fi
}

# Get a list of docker images.
readDockerInfoFromUrl(){
    while true
    do
        list_url=$DEFAULT_DOCKER_IMAGE_LISTS
        content=$(curl --connect-timeout 10 -m 10 -s $list_url)
        if [ ! -z "$content" ]; then
            break
        else
            echo -e "    ${RED}Unable to get docker image list due to network issues, try again.${PLAIN}"

            # switch sources of docker image lists
            if [ "$list_url" = "$DEFAULT_DOCKER_OFFLINE_CPU_ZH_LISTS_OSS" ]; then
                DEFAULT_DOCKER_IMAGE_LISTS=$DEFAULT_DOCKER_OFFLINE_CPU_ZH_LISTS_GIT
            else
                DEFAULT_DOCKER_IMAGE_LISTS=$DEFAULT_DOCKER_OFFLINE_CPU_ZH_LISTS_OSS
            fi
        fi
    done
    array=($(echo "$content"))
    len=${#array[@]}

    stage=0
    docker_flag="DOCKER:"
    judge_flag=":"
    for i in ${array[@]}
    do
        findTypeOfDockerInfo $i
        if [ "$docker_info_cur_key" = "DOCKER:" ]; then
            if [ ! -z "$docker_info_cur_val" ]; then
                docker_name=${DEFAULT_FUNASR_DOCKER_URL}:${docker_info_cur_val}
                DOCKER_IMAGES[${#DOCKER_IMAGES[*]}]=$docker_name 
            fi
        elif [ "$docker_info_cur_key" = "DEFAULT_ASR_MODEL:" ]; then
            if [ ! -z "$docker_info_cur_val" ]; then
                PARAMS_ASR_ID=$docker_info_cur_val
            fi
        elif [ "$docker_info_cur_key" = "DEFAULT_VAD_MODEL:" ]; then
            if [ ! -z "$docker_info_cur_val" ]; then
                PARAMS_VAD_ID=$docker_info_cur_val
            fi
        elif [ "$docker_info_cur_key" = "DEFAULT_PUNC_MODEL:" ]; then
            if [ ! -z "$docker_info_cur_val" ]; then
                PARAMS_PUNC_ID=$docker_info_cur_val
            fi
        fi
    done
    echo -e "    $DONE"
}

# Make sure root user.
rootNess(){
    echo -e "${UNDERLINE}${BOLD}[0/5]${PLAIN}"
    echo -e "  ${YELLOW}Please check root access.${PLAIN}"

    echo -e "    ${WARNING} MUST RUN AS ${RED}ROOT${PLAIN} USER!"
    if [[ $EUID -ne 0 ]]; then
        echo -e "  ${ERROR} MUST RUN AS ${RED}ROOT${PLAIN} USER!"
    fi

    check_sudo=$(which sudo | wc -l)
    if [ $check_sudo -eq 0 ]; then
        SUDO_CMD=""
    fi

    cd $CUR_DIR
    echo
}

# Get a list of docker images and select them.
selectDockerImages(){
    echo -e "${UNDERLINE}${BOLD}[1/5]${PLAIN}"
    echo -e "  ${YELLOW}Getting the list of docker images, please wait a few seconds.${PLAIN}"
    readDockerInfoFromUrl
    echo

    the_latest_docker_image=$PARAMS_DOCKER_IMAGE

    echo -e "  ${YELLOW}Please choose the Docker image.${PLAIN}"
    menuSelection ${DOCKER_IMAGES[*]}
    result=$?
    index=`expr ${result} - 1`

    PARAMS_DOCKER_IMAGE=${DOCKER_IMAGES[${index}]}
    echo -e "  ${UNDERLINE}You have chosen the Docker image:${PLAIN} ${GREEN}${PARAMS_DOCKER_IMAGE}${PLAIN}"

    if [ -z "$the_latest_docker_image" ] && [ -z "$PARAMS_FUNASR_DOCKER_ID" ]; then
        result=0
    else
        #  0: DOCKER is not running
        # 60: DOCKER_ID is empty
        # 61: DOCKER_IMAGE is empty
        # 62: DOCKER is running
        # 63: DOCKER_ID and DOCKER_IMAGE are empty
        checkDockerIdExist "install"
        result=$?
        result=`expr ${result} + 0`
        if [ $result -eq 60 ]; then
            result=0
        elif [ $result -eq 61 ]; then
            echo
            echo -e "  ${RED}Please run (${PLAIN}${GREEN}${SUDO_CMD} bash funasr-runtime-deploy-offline-cpu-zh.sh install${PLAIN}${RED}) to install Docker first.${PLAIN}"
        elif [ $result -eq 62 ]; then
            echo
            echo -e "  ${RED}Docker: ${PARAMS_DOCKER_IMAGE} ${PARAMS_FUNASR_DOCKER_ID} has been launched, please run (${PLAIN}${GREEN}${SUDO_CMD} bash funasr-runtime-deploy-offline-cpu-zh.sh remove${PLAIN}${RED}) to remove Docker first ant then install.${PLAIN}"
        elif [ $result -eq 63 ]; then
            result=0
        fi
    fi

    echo
    return $result
}

# Configure FunASR server host port setting.
setupHostPort(){
    echo -e "${UNDERLINE}${BOLD}[2/5]${PLAIN}"

    params_host_port=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ -z "$params_host_port" ]; then
        PARAMS_HOST_PORT="10095"
    else
        PARAMS_HOST_PORT=$params_host_port
    fi

    while true
    do
        echo -e "  ${YELLOW}Please input the opened port in the host used for FunASR server.${PLAIN}"
        echo -e "  Setting the opened host port [1-65535], default(${CYAN}${PARAMS_HOST_PORT}${PLAIN}): \c"
        read PARAMS_HOST_PORT

        if [ -z "$PARAMS_HOST_PORT" ]; then
            params_host_port=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
            if [ -z "$params_host_port" ]; then
                PARAMS_HOST_PORT="10095"
            else
                PARAMS_HOST_PORT=$params_host_port
            fi
        fi
        expr $PARAMS_HOST_PORT + 0 &>/dev/null
        if [ $? -eq 0 ]; then
            if [ $PARAMS_HOST_PORT -ge 1 ] && [ $PARAMS_HOST_PORT -le 65535 ]; then
                echo -e "  ${UNDERLINE}The port of the host is${PLAIN} ${GREEN}${PARAMS_HOST_PORT}${PLAIN}"
                echo -e "  ${UNDERLINE}The port in Docker for FunASR server is${PLAIN} ${GREEN}${PARAMS_DOCKER_PORT}${PLAIN}"
                break
            else
                echo -e "  ${RED}Input error, please input correct number!${PLAIN}"
            fi
        else
            echo -e "  ${RED}Input error, please input correct number!${PLAIN}"
        fi
    done
    echo
}

complementParameters(){
    # parameters about ASR model
    if [ ! -z "$PARAMS_ASR_ID" ]; then
        PARAMS_DOCKER_ASR_PATH=${PARAMS_DOWNLOAD_MODEL_DIR}/${PARAMS_ASR_ID}
        PARAMS_DOCKER_ASR_DIR=$(dirname "$PARAMS_DOCKER_ASR_PATH")
        PARAMS_LOCAL_ASR_PATH=${PARAMS_FUNASR_LOCAL_MODELS_DIR}/${PARAMS_ASR_ID}
        PARAMS_LOCAL_ASR_DIR=$(dirname "$PARAMS_LOCAL_ASR_PATH")
    fi

    # parameters about VAD model
    if [ ! -z "$PARAMS_VAD_ID" ]; then
            PARAMS_DOCKER_VAD_PATH=${PARAMS_DOWNLOAD_MODEL_DIR}/${PARAMS_VAD_ID}
            PARAMS_DOCKER_VAD_DIR=$(dirname "$PARAMS_DOCKER_VAD_PATH")
            PARAMS_LOCAL_VAD_PATH=${PARAMS_FUNASR_LOCAL_MODELS_DIR}/${PARAMS_VAD_ID}
            PARAMS_LOCAL_VAD_DIR=$(dirname "$PARAMS_LOCAL_VAD_PATH")
    fi

    # parameters about PUNC model
    if [ ! -z "$PARAMS_PUNC_ID" ]; then
        PARAMS_DOCKER_PUNC_PATH=${PARAMS_DOWNLOAD_MODEL_DIR}/${PARAMS_PUNC_ID}
        PARAMS_DOCKER_PUNC_DIR=$(dirname "${PARAMS_DOCKER_PUNC_PATH}")
        PARAMS_LOCAL_PUNC_PATH=${PARAMS_FUNASR_LOCAL_MODELS_DIR}/${PARAMS_PUNC_ID}
        PARAMS_LOCAL_PUNC_DIR=$(dirname "${PARAMS_LOCAL_PUNC_PATH}")
    fi

    # parameters about thread_num
    params_decoder_thread_num=`sed '/^PARAMS_DECODER_THREAD_NUM=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ -z "$params_decoder_thread_num" ]; then
        PARAMS_DECODER_THREAD_NUM=$CPUNUM
    else
        PARAMS_DECODER_THREAD_NUM=$params_decoder_thread_num
    fi

    multiple_io=4
    PARAMS_DECODER_THREAD_NUM=`expr ${PARAMS_DECODER_THREAD_NUM} + 0`
    PARAMS_IO_THREAD_NUM=`expr ${PARAMS_DECODER_THREAD_NUM} / ${multiple_io}`
    if [ $PARAMS_IO_THREAD_NUM -eq 0 ]; then
        PARAMS_IO_THREAD_NUM=1
    fi
}

paramsFromDefault(){
    initConfiguration

    echo -e "  ${YELLOW}Load parameters from${PLAIN} ${GREEN}${DEFAULT_FUNASR_CONFIG_FILE}${PLAIN}"
    echo

    funasr_local_workspace=`sed '/^PARAMS_FUNASR_LOCAL_WORKSPACE=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$funasr_local_workspace" ]; then
        PARAMS_FUNASR_LOCAL_WORKSPACE=$funasr_local_workspace
    fi
    funasr_samples_local_dir=`sed '/^PARAMS_FUNASR_SAMPLES_LOCAL_DIR=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$funasr_samples_local_dir" ]; then
        PARAMS_FUNASR_SAMPLES_LOCAL_DIR=$funasr_samples_local_dir
    fi
    funasr_samples_local_path=`sed '/^PARAMS_FUNASR_SAMPLES_LOCAL_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$funasr_samples_local_path" ]; then
        PARAMS_FUNASR_SAMPLES_LOCAL_PATH=$funasr_samples_local_path
    fi
    funasr_local_models_dir=`sed '/^PARAMS_FUNASR_LOCAL_MODELS_DIR=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$funasr_local_models_dir" ]; then
        PARAMS_FUNASR_LOCAL_MODELS_DIR=$funasr_local_models_dir
    fi

    docker_image=`sed '/^PARAMS_DOCKER_IMAGE=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$docker_image" ]; then
        PARAMS_DOCKER_IMAGE=$docker_image
    fi
    download_model_dir=`sed '/^PARAMS_DOWNLOAD_MODEL_DIR=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$download_model_dir" ]; then
        PARAMS_DOWNLOAD_MODEL_DIR=$download_model_dir
    fi
    PARAMS_LOCAL_ASR_PATH=`sed '/^PARAMS_LOCAL_ASR_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$local_asr_path" ]; then
        PARAMS_LOCAL_ASR_PATH=$local_asr_path
    fi
    docker_asr_path=`sed '/^PARAMS_DOCKER_ASR_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$docker_asr_path" ]; then
        PARAMS_DOCKER_ASR_PATH=$docker_asr_path
    fi
    asr_id=`sed '/^PARAMS_ASR_ID=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$asr_id" ]; then
        PARAMS_ASR_ID=$asr_id
    fi
    local_vad_path=`sed '/^PARAMS_LOCAL_VAD_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$local_vad_path" ]; then
        PARAMS_LOCAL_VAD_PATH=$local_vad_path
    fi
    docker_vad_path=`sed '/^PARAMS_DOCKER_VAD_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$docker_vad_path" ]; then
        PARAMS_DOCKER_VAD_PATH=$docker_vad_path
    fi
    vad_id=`sed '/^PARAMS_VAD_ID=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$vad_id" ]; then
        PARAMS_VAD_ID=$vad_id
    fi
    local_punc_path=`sed '/^PARAMS_LOCAL_PUNC_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$local_punc_path" ]; then
        PARAMS_LOCAL_PUNC_PATH=$local_punc_path
    fi
    docker_punc_path=`sed '/^PARAMS_DOCKER_PUNC_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$docker_punc_path" ]; then
        PARAMS_DOCKER_PUNC_PATH=$docker_punc_path
    fi
    punc_id=`sed '/^PARAMS_PUNC_ID=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$punc_id" ]; then
        PARAMS_PUNC_ID=$punc_id
    fi
    docker_exec_path=`sed '/^PARAMS_DOCKER_EXEC_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$docker_exec_path" ]; then
        PARAMS_DOCKER_EXEC_PATH=$docker_exec_path
    fi
    host_port=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$host_port" ]; then
        PARAMS_HOST_PORT=$host_port
    fi
    docker_port=`sed '/^PARAMS_DOCKER_PORT=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$docker_port" ]; then
        PARAMS_DOCKER_PORT=$docker_port
    fi
    decode_thread_num=`sed '/^PARAMS_DECODER_THREAD_NUM=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$decode_thread_num" ]; then
        PARAMS_DECODER_THREAD_NUM=$decode_thread_num
    fi
    io_thread_num=`sed '/^PARAMS_IO_THREAD_NUM=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$io_thread_num" ]; then
        PARAMS_IO_THREAD_NUM=$io_thread_num
    fi

    ssl_flag=`sed '/^PARAMS_SSL_FLAG=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$ssl_flag" ]; then
        PARAMS_SSL_FLAG=$ssl_flag
    fi
    docker_id=`sed '/^PARAMS_FUNASR_DOCKER_ID=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ ! -z "$docker_id" ]; then
        PARAMS_FUNASR_DOCKER_ID=$docker_id
    fi
}

saveParams(){
    echo "$i" > $DEFAULT_FUNASR_CONFIG_FILE
    echo -e "  ${GREEN}Parameters are stored in the file ${DEFAULT_FUNASR_CONFIG_FILE}${PLAIN}"

    echo "PARAMS_DOCKER_IMAGE=${PARAMS_DOCKER_IMAGE}" > $DEFAULT_FUNASR_CONFIG_FILE

    echo "PARAMS_FUNASR_LOCAL_WORKSPACE=${PARAMS_FUNASR_LOCAL_WORKSPACE}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_FUNASR_SAMPLES_LOCAL_DIR=${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_FUNASR_SAMPLES_LOCAL_PATH=${PARAMS_FUNASR_SAMPLES_LOCAL_PATH}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_FUNASR_LOCAL_MODELS_DIR=${PARAMS_FUNASR_LOCAL_MODELS_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE

    echo "PARAMS_DOWNLOAD_MODEL_DIR=${PARAMS_DOWNLOAD_MODEL_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE

    echo "PARAMS_DOCKER_EXEC_PATH=${PARAMS_DOCKER_EXEC_PATH}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_EXEC_DIR=${PARAMS_DOCKER_EXEC_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE

    echo "PARAMS_LOCAL_ASR_PATH=${PARAMS_LOCAL_ASR_PATH}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_LOCAL_ASR_DIR=${PARAMS_LOCAL_ASR_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_ASR_PATH=${PARAMS_DOCKER_ASR_PATH}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_ASR_DIR=${PARAMS_DOCKER_ASR_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_ASR_ID=${PARAMS_ASR_ID}" >> $DEFAULT_FUNASR_CONFIG_FILE

    echo "PARAMS_LOCAL_PUNC_PATH=${PARAMS_LOCAL_PUNC_PATH}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_LOCAL_PUNC_DIR=${PARAMS_LOCAL_PUNC_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_PUNC_PATH=${PARAMS_DOCKER_PUNC_PATH}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_PUNC_DIR=${PARAMS_DOCKER_PUNC_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_PUNC_ID=${PARAMS_PUNC_ID}" >> $DEFAULT_FUNASR_CONFIG_FILE

    echo "PARAMS_LOCAL_VAD_PATH=${PARAMS_LOCAL_VAD_PATH}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_LOCAL_VAD_DIR=${PARAMS_LOCAL_VAD_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_VAD_PATH=${PARAMS_DOCKER_VAD_PATH}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_VAD_DIR=${PARAMS_DOCKER_VAD_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_VAD_ID=${PARAMS_VAD_ID}" >> $DEFAULT_FUNASR_CONFIG_FILE

    echo "PARAMS_HOST_PORT=${PARAMS_HOST_PORT}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_PORT=${PARAMS_DOCKER_PORT}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_DECODER_THREAD_NUM=${PARAMS_DECODER_THREAD_NUM}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_IO_THREAD_NUM=${PARAMS_IO_THREAD_NUM}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_SSL_FLAG=${PARAMS_SSL_FLAG}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_FUNASR_DOCKER_ID=${PARAMS_FUNASR_DOCKER_ID}" >> $DEFAULT_FUNASR_CONFIG_FILE

    serverConfigGeneration
    echo "${daemon_server_config}" > $DEFAULT_FUNASR_SERVER_CONFIG_FILE
}

showAllParams(){
    echo -e "${UNDERLINE}${BOLD}[3/5]${PLAIN}"
    echo -e "  ${YELLOW}Show parameters of FunASR server setting and confirm to run ...${PLAIN}"
    echo

    only_show_flag=$1

    if [ ! -z "$PARAMS_DOCKER_IMAGE" ]; then
        echo -e "  The current Docker image is                                    : ${GREEN}${PARAMS_DOCKER_IMAGE}${PLAIN}"
    fi
    if [ ! -z "$PARAMS_FUNASR_LOCAL_WORKSPACE" ]; then
        echo -e "  The local workspace path is                                    : ${GREEN}${PARAMS_FUNASR_LOCAL_WORKSPACE}${PLAIN}"
    fi

    if [ ! -z "$PARAMS_DOWNLOAD_MODEL_DIR" ]; then
        echo -e "  The model will be automatically downloaded in Docker           : ${GREEN}${PARAMS_DOWNLOAD_MODEL_DIR}${PLAIN}"
    fi
    echo

    if [ ! -z "$PARAMS_ASR_ID" ]; then
        echo -e "  The ASR model_id used                                          : ${GREEN}${PARAMS_ASR_ID}${PLAIN}"
    fi
    if [ ! -z "$PARAMS_LOCAL_ASR_PATH" ]; then
        echo -e "  The path to the local ASR model directory for the load         : ${GREEN}${PARAMS_LOCAL_ASR_PATH}${PLAIN}"
    fi
    echo -e "  The ASR model directory corresponds to the directory in Docker : ${GREEN}${PARAMS_DOCKER_ASR_PATH}${PLAIN}"

    if [ ! -z "$PARAMS_VAD_ID" ]; then
        echo -e "  The VAD model_id used                                          : ${GREEN}${PARAMS_VAD_ID}${PLAIN}"
    fi
    if [ ! -z "$PARAMS_LOCAL_VAD_PATH" ]; then
        echo -e "  The path to the local VAD model directory for the load         : ${GREEN}${PARAMS_LOCAL_VAD_PATH}${PLAIN}"
    fi
    echo -e "  The VAD model directory corresponds to the directory in Docker : ${GREEN}${PARAMS_DOCKER_VAD_PATH}${PLAIN}"

    if [ ! -z "$PARAMS_PUNC_ID" ]; then
        echo -e "  The PUNC model_id used                                         : ${GREEN}${PARAMS_PUNC_ID}${PLAIN}"
    fi
    if [ ! -z "$PARAMS_LOCAL_PUNC_PATH" ]; then
        echo -e "  The path to the local PUNC model directory for the load        : ${GREEN}${PARAMS_LOCAL_PUNC_PATH}${PLAIN}"
    fi
    echo -e "  The PUNC model directory corresponds to the directory in Docker: ${GREEN}${PARAMS_DOCKER_PUNC_PATH}${PLAIN}"
    echo

    echo -e "  The path in the docker of the FunASR service executor          : ${GREEN}${PARAMS_DOCKER_EXEC_PATH}${PLAIN}"

    echo -e "  Set the host port used for use by the FunASR service           : ${GREEN}${PARAMS_HOST_PORT}${PLAIN}"
    echo -e "  Set the docker port used by the FunASR service                 : ${GREEN}${PARAMS_DOCKER_PORT}${PLAIN}"

    echo -e "  Set the number of threads used for decoding the FunASR service : ${GREEN}${PARAMS_DECODER_THREAD_NUM}${PLAIN}"
    echo -e "  Set the number of threads used for IO the FunASR service       : ${GREEN}${PARAMS_IO_THREAD_NUM}${PLAIN}"
    echo

    if [ ! -z "$PARAMS_FUNASR_SAMPLES_LOCAL_DIR" ]; then
        echo -e "  Sample code will be store in local                             : ${GREEN}${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}${PLAIN}"
    fi
    if [ ! -z "$PARAMS_SSL_FLAG" ]; then
        echo -e "  The flag for the use of SSL                                    : ${GREEN}${PARAMS_SSL_FLAG}${PLAIN}"
    fi
    if [ "$only_show_flag" = "only_show" ] && [ ! -z "$PARAMS_FUNASR_DOCKER_ID" ]; then
        echo -e "  The docker ID that already exists is                           : ${GREEN}${PARAMS_FUNASR_DOCKER_ID}${PLAIN}"
    fi

    echo

    if [ "$only_show_flag" = "only_show" ]; then
        return 0
    fi

    while true
    do
        params_confirm="y"
        echo -e "  ${YELLOW}Please input [Y/n] to confirm the parameters.${PLAIN}"
        echo -e "  [y] Verify that these parameters are correct and that the service will run."
        echo -e "  [n] The parameters set are incorrect, it will be rolled out, please rerun."
        echo -e "  read confirmation[${CYAN}Y${PLAIN}/n]: \c"
        read params_confirm

        if [ -z "$params_confirm" ]; then
            params_confirm="y"
        fi
        YES="Y"
        yes="y"
        NO="N"
        no="n"
        echo
        if [ "$params_confirm" = "$YES" ] || [ "$params_confirm" = "$yes" ]; then
            echo -e "  ${GREEN}Will run FunASR server later ...${PLAIN}"
            break
        elif [ "$params_confirm" = "$NO" ] || [ "$params_confirm" = "$no" ]; then
            echo -e "  ${RED}The parameters set are incorrect, please rerun ...${PLAIN}"
            exit 1
        else
            echo "again ..."
        fi
    done

    saveParams
    echo
    sleep 1
}

# Install docker
installFunasrDocker(){
    echo -e "${UNDERLINE}${BOLD}[4/5]${PLAIN}"

    if [ $DOCKERINFOLEN -gt 30 ]; then
        echo -e "  ${YELLOW}Docker has installed.${PLAIN}"
    else
        lowercase_osid=$(echo ${OSID} | tr '[A-Z]' '[a-z]')
        echo -e "  ${YELLOW}Start install docker for ${lowercase_osid} ${PLAIN}"
        DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun"
        DOCKER_INSTALL_RUN_CMD=""

        case "$lowercase_osid" in
            ubuntu)
                DOCKER_INSTALL_CMD="curl -fsSL https://test.docker.com -o test-docker.sh"
                DOCKER_INSTALL_RUN_CMD="${SUDO_CMD} sh test-docker.sh"
                ;;
            centos)
                DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun"
                ;;
            debian)
                DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com -o get-docker.sh"
                DOCKER_INSTALL_RUN_CMD="${SUDO_CMD} sh get-docker.sh"
                ;;
            \"alios\")
                DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com -o get-docker.sh"
                DOCKER_INSTALL_RUN_CMD="${SUDO_CMD} sh get-docker.sh"
                ;;
            \"alinux\")
                DOCKER_INSTALL_CMD="${SUDO_CMD} yum -y install dnf"
                DOCKER_INSTALL_RUN_CMD="${SUDO_CMD} dnf -y install docker"
                ;;
            *)
                echo -e "  ${RED}$lowercase_osid is not supported.${PLAIN}"
                ;;
        esac

        echo -e "  Get docker installer: ${GREEN}${DOCKER_INSTALL_CMD}${PLAIN}"
        echo -e "  Get docker run: ${GREEN}${DOCKER_INSTALL_RUN_CMD}${PLAIN}"

        $DOCKER_INSTALL_CMD
        if [ ! -z "$DOCKER_INSTALL_RUN_CMD" ]; then
            $DOCKER_INSTALL_RUN_CMD
        fi
        $SUDO_CMD systemctl start docker

        DOCKERINFO=$(${SUDO_CMD} docker info | wc -l)
        DOCKERINFOLEN=`expr ${DOCKERINFO} + 0`
        if [ $DOCKERINFOLEN -gt 30 ]; then
            echo -e "  ${GREEN}Docker install success, start docker server.${PLAIN}"
            $SUDO_CMD systemctl start docker
        else
            echo -e "  ${RED}Docker install failed!${PLAIN}"
            exit 1
        fi
    fi

    echo
    sleep 1

    # Download docker image
    echo -e "  ${YELLOW}Pull docker image(${PARAMS_DOCKER_IMAGE})...${PLAIN}"

    ${SUDO_CMD} docker pull $PARAMS_DOCKER_IMAGE

    echo
    sleep 1
}

dockerRun(){
    echo -e "${UNDERLINE}${BOLD}[5/5]${PLAIN}"
    echo -e "  ${YELLOW}Construct command and run docker ...${PLAIN}"

    start_flag=$1
    if [ "$start_flag" = "install" ]; then
        run_cmd="${SUDO_CMD} docker run"
        port_map=" -p ${PARAMS_HOST_PORT}:${PARAMS_DOCKER_PORT}"
        env_params=" --privileged=true"
        dir_map_params=" -v ${DEFAULT_FUNASR_CONFIG_DIR}:/workspace/.config -v ${PARAMS_FUNASR_LOCAL_MODELS_DIR}:${PARAMS_DOWNLOAD_MODEL_DIR}"

        serverConfigGeneration
        env_params=" ${env_params} --env DAEMON_SERVER_CONFIG=${daemon_server_config}"

        run_cmd="${run_cmd}${port_map}${dir_map_params}${env_params}"
        run_cmd="${run_cmd} -it -d ${PARAMS_DOCKER_IMAGE}"
    else
        #  0: DOCKER is not running
        # 60: DOCKER_ID is empty
        # 61: DOCKER_IMAGE is empty
        # 62: DOCKER is running
        checkDockerIdExist $start_flag
        result=$?
        result=`expr ${result} + 0`
        if [ $result -eq 60 ]; then
            echo
            echo -e "  ${RED}Please run (${PLAIN}${GREEN}${SUDO_CMD} bash funasr-runtime-deploy-offline-cpu-zh.sh install${PLAIN}${RED}) to install Docker first.${PLAIN}"
            return $result
        elif [ $result -eq 61 ]; then
            echo
            echo -e "  ${RED}Please run (${PLAIN}${GREEN}${SUDO_CMD} bash funasr-runtime-deploy-offline-cpu-zh.sh install${PLAIN}${RED}) to install Docker first.${PLAIN}"
            return $result
        elif [ $result -eq 62 ]; then
            echo
            echo -e "  ${RED}Docker: ${PARAMS_DOCKER_IMAGE} ${PARAMS_FUNASR_DOCKER_ID} has been launched, please run (${PLAIN}${GREEN}${SUDO_CMD} bash funasr-runtime-deploy-offline-cpu-zh.sh stop${PLAIN}${RED}) to stop Docker first.${PLAIN}"
            return $result
        fi

        run_cmd="${SUDO_CMD} docker restart ${PARAMS_FUNASR_DOCKER_ID}"
    fi

    rm -f ${DEFAULT_FUNASR_PROGRESS_TXT}
    rm -f ${DEFAULT_FUNASR_SERVER_LOG}

    $run_cmd

    echo
    echo -e "  ${YELLOW}Loading models:${PLAIN}"

    getDockerId
    saveParams

    # Hide the cursor, start draw progress.
    printf "\e[?25l"
    while true
    do
        serverProgress
        result=$?
        stage=`expr ${result} + 0`
        if [ ${stage} -eq 0 ]; then
            break
        elif [ ${stage} -gt 0 ] && [ ${stage} -lt 6 ]; then
            sleep 0.1
            # clear 3 lines
            printf "\033[3A"
        elif [ ${stage} -eq 6 ]; then
            break
        elif [ ${stage} -eq 98 ]; then
            return 98
        else
            echo -e "  ${RED}Starting FunASR server failed.${PLAIN}"
            echo
            # Display the cursor
            printf "\e[?25h"
            return 99
        fi
    done
    # Display the cursor
    printf "\e[?25h"

    echo -e "  ${GREEN}The service has been started.${PLAIN}"
    echo

    deploySamples
    echo -e "  ${BOLD}The sample code is already stored in the ${PLAIN}(${GREEN}${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}${PLAIN}) ."
    echo -e "  ${BOLD}If you want to see an example of how to use the client, you can run ${PLAIN}${GREEN}${SUDO_CMD} bash funasr-runtime-deploy-offline-cpu-zh.sh client${PLAIN} ."
    echo
}

daemon_server_config=""
serverConfigGeneration(){
    # params about models
	if [ ! -z "$PARAMS_ASR_ID" ]; then
        asr_params="\"--model-dir\":\"${PARAMS_ASR_ID}\""
    else
        if [ ! -z "$PARAMS_LOCAL_ASR_PATH" ]; then
            dir_map_params="${dir_map_params} -v ${PARAMS_LOCAL_ASR_PATH}:${PARAMS_DOCKER_ASR_PATH}"
        fi
        asr_params="\"--model-dir\":\"${PARAMS_DOCKER_ASR_PATH}\""
    fi
    if [ ! -z "$PARAMS_VAD_ID" ]; then
        vad_params="\"--vad-dir\":\"${PARAMS_VAD_ID}\""
    else
        if [ ! -z "$PARAMS_LOCAL_VAD_PATH" ]; then
            dir_map_params="${dir_map_params} -v ${PARAMS_LOCAL_VAD_PATH}:${PARAMS_DOCKER_VAD_PATH}"
        fi
        vad_params="\"--vad-dir\":\"${PARAMS_DOCKER_VAD_PATH}\""
    fi
    if [ ! -z "$PARAMS_PUNC_ID" ]; then
        punc_params="\"--punc-dir\":\"${PARAMS_PUNC_ID}\""
    else
        if [ ! -z "$PARAMS_LOCAL_PUNC_PATH" ]; then
            dir_map_params="${dir_map_params} -v ${PARAMS_LOCAL_VAD_PATH}:${PARAMS_DOCKER_VAD_PATH}"
        fi
        punc_params="\"--punc-dir\":\"${PARAMS_DOCKER_PUNC_PATH}\""
    fi
    download_params="\"--download-model-dir\":\"${PARAMS_DOWNLOAD_MODEL_DIR}\""
    model_params="${asr_params},${vad_params},${punc_params},${download_params}"

    # params about thread_num
    decoder_params="\"--decoder-thread-num\":\"${PARAMS_DECODER_THREAD_NUM}\""
    io_params="\"--io-thread-num\":\"${PARAMS_IO_THREAD_NUM}\""
    thread_params=${decoder_params},${io_params}

    # params about port and ssl
    port_params="\"--port\":\"${PARAMS_DOCKER_PORT}\""
    if [ $PARAMS_SSL_FLAG -eq 0 ]; then
        crt_path="\"--certfile\":\"\""
        key_path="\"--keyfile\":\"\""
    else
        crt_path="\"--certfile\":\"/workspace/FunASR/funasr/runtime/ssl_key/server.crt\""
        key_path="\"--keyfile\":\"/workspace/FunASR/funasr/runtime/ssl_key/server.key\""
    fi

    exec_params="\"exec\":\"${PARAMS_DOCKER_EXEC_PATH}\""

    daemon_server_config="{\"server\":[{${exec_params},${model_params},${thread_params},${port_params},${crt_path},${key_path}}]}"
}

installPythonDependencyForPython(){
    echo -e "${YELLOW}Install Python dependent environments ...${PLAIN}"

    echo -e "  Export dependency of Cpp sample."
    pre_cmd="export LD_LIBRARY_PATH=${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}/cpp/libs:\$LD_LIBRARY_PATH"
    $pre_cmd
    echo

    echo -e "  Install requirements of Python sample."
    pre_cmd="pip3 install click>=8.0.4"
    $pre_cmd
    echo

    pre_cmd="pip3 install -r ${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}/python/requirements_client.txt"
    echo -e "  Run ${BLUE}${pre_cmd}${PLAIN}"
    $pre_cmd
    echo

    lowercase_osid=$(echo ${OSID} | tr '[A-Z]' '[a-z]')
    case "$lowercase_osid" in
        ubuntu)
            pre_cmd="${SUDO_CMD} apt-get install -y ffmpeg"
            ;;
        centos)
            pre_cmd="${SUDO_CMD} yum install -y ffmpeg"
            ;;
        debian)
            pre_cmd="${SUDO_CMD} apt-get install -y ffmpeg"
            ;;
        \"alios\")
            pre_cmd="${SUDO_CMD} yum install -y ffmpeg"
            ;;
        \"alinux\")
            pre_cmd="${SUDO_CMD} yum install -y ffmpeg"
            ;;
        *)
            echo -e "  ${RED}$lowercase_osid is not supported.${PLAIN}"
            ;;
    esac
    echo -e "  Run ${BLUE}${pre_cmd}${PLAIN}"
    echo

    pre_cmd="pip3 install ffmpeg-python"
    echo -e "  Run ${BLUE}${pre_cmd}${PLAIN}"
    $pre_cmd
    echo
}

deploySamples(){
    if [ ! -d $PARAMS_FUNASR_SAMPLES_LOCAL_DIR ]; then
        echo -e "${YELLOW}Downloading samples to${PLAIN} ${CYAN}${PARAMS_FUNASR_LOCAL_WORKSPACE}${PLAIN} ${YELLOW}...${PLAIN}"

        download_cmd="curl ${DEFAULT_SAMPLES_URL} -o ${PARAMS_FUNASR_SAMPLES_LOCAL_PATH}"
        untar_cmd="tar -zxf ${PARAMS_FUNASR_SAMPLES_LOCAL_PATH} -C ${PARAMS_FUNASR_LOCAL_WORKSPACE}"

        if [ ! -f "$PARAMS_FUNASR_SAMPLES_LOCAL_PATH" ]; then
            $download_cmd
        fi
        $untar_cmd
        echo

        installPythonDependencyForPython
        echo
    fi
}

getDockerId(){
    id=""
    array=($(${SUDO_CMD} docker ps -a | grep ${PARAMS_DOCKER_IMAGE} | awk '{print $1}'))
    len=${#array[@]}
    if [ $len -ge 1 ]; then
        # get the first id
        id=$array
        if [ ! -z "$id" ]; then
            PARAMS_FUNASR_DOCKER_ID=$id
        fi
    fi
}

checkDockerImageExist(){
    result=1
    if [ -z "$PARAMS_DOCKER_IMAGE" ]; then
        return 50
    else
        result=$(${SUDO_CMD} docker ps | grep ${PARAMS_DOCKER_IMAGE} | wc -l)
    fi
    result=`expr ${result} + 0`
    echo "checkDockerImageExist result0: " $result
    if [ $result -ne 0 ]; then
        # found docker
        return 51
    else
        return 0
    fi
}

checkDockerIdExist(){
    result=0
    if [ -z "$PARAMS_FUNASR_DOCKER_ID" ]; then
        if [ -z "$PARAMS_DOCKER_IMAGE" ]; then
            return 63
        else
            return 60
        fi
    else
        if [ -z "$PARAMS_DOCKER_IMAGE" ]; then
            return 61
        else
            if [ "$1" = "install" ]; then
                result=$(${SUDO_CMD} docker ps -a | grep ${PARAMS_DOCKER_IMAGE} | grep ${PARAMS_FUNASR_DOCKER_ID} | wc -l)
            else
                result=$(${SUDO_CMD} docker ps | grep ${PARAMS_DOCKER_IMAGE} | grep ${PARAMS_FUNASR_DOCKER_ID} | wc -l)
            fi
        fi
    fi
    result=`expr ${result} + 0`
    if [ $result -eq 1 ]; then
        # found docker
        return 62
    else
        return 0
    fi
}

dockerStop(){
    if [ -z "$PARAMS_FUNASR_DOCKER_ID" ]; then
        echo -e "  ${RED}DOCKER_ID is empty, cannot stop docker.${PLAIN}"
    else
        echo -e "  ${YELLOW}Stop docker(${PLAIN}${GREEN}${PARAMS_DOCKER_IMAGE} ${PARAMS_FUNASR_DOCKER_ID}${PLAIN}${YELLOW}) server ...${PLAIN}"
        ${SUDO_CMD} docker stop ${PARAMS_FUNASR_DOCKER_ID}
    fi
    echo
}

dockerRemove(){
    if [ -z "$PARAMS_FUNASR_DOCKER_ID" ]; then
        echo -e "  ${RED}DOCKER_ID is empty, cannot remove docker.${PLAIN}"
    else
        echo -e "  ${YELLOW}Remove docker(${PLAIN}${GREEN}${PARAMS_DOCKER_IMAGE} ${PARAMS_FUNASR_DOCKER_ID}${PLAIN}${YELLOW}) ...${PLAIN}"
        ${SUDO_CMD} docker rm ${PARAMS_FUNASR_DOCKER_ID}
    fi

    echo
}

modelChange(){
    model_type=$1
    model_id=$2
    local_flag=0

    relativePathToFullPath $model_id
    if [ -d "$full_path" ]; then
        local_flag=1
        model_id=$full_path
    else
        local_flag=0
    fi
    full_path=""

    result=$(echo ${model_type} | grep "\-\-asr_model")
    if [ "$result" != "" ]; then
        if [ $local_flag -eq 0 ]; then
            PARAMS_ASR_ID=$model_id
            PARAMS_DOCKER_ASR_PATH=${PARAMS_DOWNLOAD_MODEL_DIR}/${PARAMS_ASR_ID}
            PARAMS_DOCKER_ASR_DIR=$(dirname "${PARAMS_DOCKER_ASR_PATH}")
            PARAMS_LOCAL_ASR_PATH=${PARAMS_FUNASR_LOCAL_MODELS_DIR}/${PARAMS_ASR_ID}
            PARAMS_LOCAL_ASR_DIR=$(dirname "${PARAMS_LOCAL_ASR_PATH}")
        else
            PARAMS_ASR_ID=""
            PARAMS_LOCAL_ASR_PATH=$model_id
            if [ ! -d "$PARAMS_LOCAL_ASR_PATH" ]; then
                echo -e "  ${RED}${PARAMS_LOCAL_ASR_PATH} does not exist, please set again.${PLAIN}"
            else
                model_name=$(basename "${PARAMS_LOCAL_ASR_PATH}")
                PARAMS_LOCAL_ASR_DIR=$(dirname "${PARAMS_LOCAL_ASR_PATH}")
                PARAMS_DOCKER_ASR_DIR=$PARAMS_DOWNLOAD_MODEL_DIR
                PARAMS_DOCKER_ASR_PATH=${PARAMS_DOCKER_ASR_DIR}/${model_name}
            fi
        fi
    fi
    result=$(echo ${model_type} | grep "\-\-vad_model")
    if [ "$result" != "" ]; then
        if [ $local_flag -eq 0 ]; then
            PARAMS_VAD_ID=$model_id
            PARAMS_DOCKER_VAD_PATH=${PARAMS_DOWNLOAD_MODEL_DIR}/${PARAMS_VAD_ID}
            PARAMS_DOCKER_VAD_DIR=$(dirname "${PARAMS_DOCKER_VAD_PATH}")
            PARAMS_LOCAL_VAD_PATH=${PARAMS_FUNASR_LOCAL_MODELS_DIR}/${PARAMS_VAD_ID}
            PARAMS_LOCAL_VAD_DIR=$(dirname "${PARAMS_LOCAL_VAD_PATH}")
        else
            PARAMS_VAD_ID=""
            PARAMS_LOCAL_VAD_PATH=$model_id
            if [ ! -d "$PARAMS_LOCAL_VAD_PATH" ]; then
                echo -e "  ${RED}${PARAMS_LOCAL_VAD_PATH} does not exist, please set again.${PLAIN}"
            else
                model_name=$(basename "${PARAMS_LOCAL_VAD_PATH}")
                PARAMS_LOCAL_VAD_DIR=$(dirname "${PARAMS_LOCAL_VAD_PATH}")
                PARAMS_DOCKER_VAD_DIR=$PARAMS_DOWNLOAD_MODEL_DIR
                PARAMS_DOCKER_VAD_PATH=${PARAMS_DOCKER_VAD_DIR}/${model_name}
            fi
        fi
    fi
    result=$(echo ${model_type} | grep "\-\-punc_model")
    if [ "$result" != "" ]; then
        if [ $local_flag -eq 0 ]; then
            PARAMS_PUNC_ID=$model_id
            PARAMS_DOCKER_PUNC_PATH=${PARAMS_DOWNLOAD_MODEL_DIR}/${PARAMS_PUNC_ID}
            PARAMS_DOCKER_PUNC_DIR=$(dirname "${PARAMS_DOCKER_PUNC_PATH}")
            PARAMS_LOCAL_PUNC_PATH=${PARAMS_FUNASR_LOCAL_MODELS_DIR}/${PARAMS_PUNC_ID}
            PARAMS_LOCAL_PUNC_DIR=$(dirname "${PARAMS_LOCAL_PUNC_PATH}")
        else
            model_name=$(basename "${PARAMS_LOCAL_PUNC_PATH}")
            PARAMS_LOCAL_PUNC_DIR=$(dirname "${PARAMS_LOCAL_PUNC_PATH}")
            PARAMS_DOCKER_PUNC_DIR=$PARAMS_DOWNLOAD_MODEL_DIR
            PARAMS_DOCKER_PUNC_PATH=${PARAMS_DOCKER_PUNC_DIR}/${model_name}
        fi
    fi
}

threadNumChange() {
    type=$1
    val=$2

    if [ -z "$val"]; then
        num=`expr ${val} + 0`
        if [ $num -ge 1 ] && [ $num -le 1024 ]; then
            result=$(echo ${type} | grep "\-\-decode_thread_num")
            if [ "$result" != "" ]; then
                PARAMS_DECODER_THREAD_NUM=$num
            fi
            result=$(echo ${type} | grep "\-\-io_thread_num")
            if [ "$result" != "" ]; then
                PARAMS_IO_THREAD_NUM=$num
            fi
        fi
    fi
}

portChange() {
    type=$1
    val=$2

    if [ ! -z "$val" ]; then
        port=`expr ${val} + 0`
        if [ $port -ge 1 ] && [ $port -le 65536 ]; then
            result=$(echo ${type} | grep "host_port")
            if [ "$result" != "" ]; then
                PARAMS_HOST_PORT=$port
            fi
            result=$(echo ${type} | grep "docker_port")
            if [ "$result" != "" ]; then
                PARAMS_DOCKER_PORT=$port
            fi
        fi
    fi
}

sampleClientRun(){
    echo -e "${YELLOW}Will download sample tools for the client to show how speech recognition works.${PLAIN}"

    download_cmd="curl ${DEFAULT_SAMPLES_URL} -o ${PARAMS_FUNASR_SAMPLES_LOCAL_PATH}"
    untar_cmd="tar -zxf ${PARAMS_FUNASR_SAMPLES_LOCAL_PATH} -C ${PARAMS_FUNASR_LOCAL_WORKSPACE}"

    if [ ! -f "$PARAMS_FUNASR_SAMPLES_LOCAL_PATH" ]; then
        $download_cmd
    fi    
    if [ -f "$PARAMS_FUNASR_SAMPLES_LOCAL_PATH" ]; then
        $untar_cmd
    fi
    if [ -d "$PARAMS_FUNASR_SAMPLES_LOCAL_DIR" ]; then
        echo -e "  Please select the client you want to run."
        menuSelection ${SAMPLE_CLIENTS[*]}
        result=$?
        index=`expr ${result} - 1`
        lang=${SAMPLE_CLIENTS[${index}]}
        echo

        server_ip="127.0.0.1"
        echo -e "  Please enter the IP of server, default(${CYAN}${server_ip}${PLAIN}): \c"
        read server_ip
        if [ -z "$server_ip" ]; then
            server_ip="127.0.0.1"
        fi

        host_port=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
        if [ -z "$host_port" ]; then
            host_port="10095"
        fi
        echo -e "  Please enter the port of server, default(${CYAN}${host_port}${PLAIN}): \c"
        read host_port
        if [ -z "$host_port" ]; then
            host_port=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
            if [ -z "$host_port" ]; then
                host_port="10095"
            fi
        fi

        wav_path="${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}/audio/asr_example.wav"
        echo -e "  Please enter the audio path, default(${CYAN}${wav_path}${PLAIN}): \c"
        read WAV_PATH
        if [ -z "$wav_path" ]; then
            wav_path="${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}/audio/asr_example.wav"
        fi

        echo
        pre_cmd=”“
        case "$lang" in
            Linux_Cpp)
                client_exec="${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}/cpp/funasr-wss-client"
                run_cmd="${client_exec} --server-ip ${server_ip} --port ${host_port} --wav-path ${wav_path}"
                ;;
            Python)
                client_exec="${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}/python/wss_client_asr.py"
                run_cmd="python3 ${client_exec} --host ${server_ip} --port ${host_port} --mode offline --audio_in ${wav_path} --send_without_sleep --output_dir ${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}/python"
                pre_cmd="pip3 install click>=8.0.4"
                echo -e "  Run ${BLUE}${pre_cmd}${PLAIN}"
                $pre_cmd
                echo
                pre_cmd="pip3 install -r ${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}/python/requirements_client.txt"
                echo -e "  Run ${BLUE}${pre_cmd}${PLAIN}"
                $pre_cmd
                echo
                ;;
            *)
                echo "${lang} is not supported."
                ;;
        esac

        echo -e "  Run ${BLUE}${run_cmd}${PLAIN}"
        $run_cmd
        echo
        echo -e "  If failed, you can try (${GREEN}${run_cmd}${PLAIN}) in your Shell."
        echo
    fi
}

paramsConfigure(){
    initConfiguration
    initParameters
    selectDockerImages
    result=$?
    result=`expr ${result} + 0`
    if [ $result -ne 0 ]; then
        return $result
    fi

    setupHostPort
    complementParameters
    return 0
}

# Display Help info
displayHelp(){
    echo -e "${UNDERLINE}Usage${PLAIN}:"
    echo -e "  $0 [OPTIONAL FLAGS]"
    echo
    echo -e "funasr-runtime-deploy-offline-cpu-zh.sh - a Bash script to install&run FunASR docker."
    echo
    echo -e "${UNDERLINE}Options${PLAIN}:"
    echo -e "   ${BOLD}-i, install, --install${PLAIN}    Install and run FunASR docker."
    echo -e "                install [--workspace] <workspace in local>"
    echo -e "                install [--ssl] <0: close SSL; 1: open SSL, default:1>"
    echo -e "   ${BOLD}-s, start  , --start${PLAIN}      Run FunASR docker with configuration that has already been set."
    echo -e "   ${BOLD}-p, stop   , --stop${PLAIN}       Stop FunASR docker."
    echo -e "   ${BOLD}-m, remove , --remove${PLAIN}     Remove FunASR docker installed."
    echo -e "   ${BOLD}-r, restart, --restart${PLAIN}    Restart FunASR docker."
    echo -e "   ${BOLD}-u, update , --update${PLAIN}     Update parameters that has already been set."
    echo -e "                update [--workspace] <workspace in local>"
    echo -e "                update [--asr_model | --vad_model | --punc_model] <model_id or local model path>"
    echo -e "                update [--host_port | --docker_port] <port number>"
    echo -e "                update [--decode_thread_num | io_thread_num] <the number of threads>"
    echo -e "                update [--ssl] <0: close SSL; 1: open SSL, default:1>"
    echo -e "   ${BOLD}-c, client , --client${PLAIN}     Get a client example to show how to initiate speech recognition."
    echo -e "   ${BOLD}-o, show   , --show${PLAIN}       Displays all parameters that have been set."
    echo -e "   ${BOLD}-v, version, --version${PLAIN}    Display current script version."
    echo -e "   ${BOLD}-h, help   , --help${PLAIN}       Display this help."
    echo
    echo -e "   Version    : ${scriptVersion} "
    echo -e "   Modify Date: ${scriptDate}"
}

parseInput(){
    local menu
    menu=($(echo "$@"))
    len=${#menu[@]}

    stage=""
    if [ $len -ge 2 ]; then
        for val in ${menu[@]}
        do
            result=$(echo $val | grep "\-\-")
            if [ "$result" != "" ]; then
                stage=$result
            else
                if [ "$stage" = "--workspace" ]; then
                    relativePathToFullPath $val
                    PARAMS_FUNASR_LOCAL_WORKSPACE=$full_path
                    full_path=""
                    if [ ! -z "$PARAMS_FUNASR_LOCAL_WORKSPACE" ]; then
                        mkdir -p $PARAMS_FUNASR_LOCAL_WORKSPACE
                    fi
                elif [ "$stage" = "--ssl" ]; then
                    PARAMS_SSL_FLAG=`expr ${val} + 0`
                fi
            fi
        done
    fi
}

# OS
OSID=$(grep ^ID= /etc/os-release | cut -d= -f2)
OSVER=$(lsb_release -cs)
OSNUM=$(grep -oE  "[0-9.]+" /etc/issue)
CPUNUM=$(cat /proc/cpuinfo | grep "processor"|wc -l)
DOCKERINFO=$(${SUDO_CMD} docker info | wc -l)
DOCKERINFOLEN=`expr ${DOCKERINFO} + 0`

# PARAMS
#  The workspace for FunASR in local
PARAMS_FUNASR_LOCAL_WORKSPACE=$DEFAULT_FUNASR_LOCAL_WORKSPACE
#  The dir stored sample code in local
PARAMS_FUNASR_SAMPLES_LOCAL_DIR=${PARAMS_FUNASR_LOCAL_WORKSPACE}/${DEFAULT_SAMPLES_DIR}
#  The path of sample code in local
PARAMS_FUNASR_SAMPLES_LOCAL_PATH=${PARAMS_FUNASR_LOCAL_WORKSPACE}/${DEFAULT_SAMPLES_NAME}.tar.gz
#  The dir stored models in local
PARAMS_FUNASR_LOCAL_MODELS_DIR="${PARAMS_FUNASR_LOCAL_WORKSPACE}/models"
#  The id of started docker
PARAMS_FUNASR_DOCKER_ID=""

#  The server excutor in local
PARAMS_DOCKER_EXEC_PATH=$DEFAULT_DOCKER_EXEC_PATH
#  The dir stored server excutor in docker
PARAMS_DOCKER_EXEC_DIR=$DEFAULT_DOCKER_EXEC_DIR

#  The dir for downloading model in docker
PARAMS_DOWNLOAD_MODEL_DIR=$DEFAULT_FUNASR_WORKSPACE_DIR
#  The Docker image name
PARAMS_DOCKER_IMAGE=""

#  The dir stored punc model in local
PARAMS_LOCAL_PUNC_DIR=""
#  The path of punc model in local
PARAMS_LOCAL_PUNC_PATH=""
#  The dir stored punc model in docker
PARAMS_DOCKER_PUNC_DIR=""
#  The path of punc model in docker
PARAMS_DOCKER_PUNC_PATH=""
#  The punc model ID in ModelScope
PARAMS_PUNC_ID="damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx"

#  The dir stored vad model in local
PARAMS_LOCAL_VAD_DIR=""
#  The path of vad model in local
PARAMS_LOCAL_VAD_PATH=""
#  The dir stored vad model in docker
PARAMS_DOCKER_VAD_DIR=""
#  The path of vad model in docker
PARAMS_DOCKER_VAD_PATH=""
#  The vad model ID in ModelScope
PARAMS_VAD_ID="damo/speech_fsmn_vad_zh-cn-16k-common-onnx"

#  The dir stored asr model in local
PARAMS_LOCAL_ASR_DIR=""
#  The path of asr model in local
PARAMS_LOCAL_ASR_PATH=""
#  The dir stored asr model in docker
PARAMS_DOCKER_ASR_DIR=""
#  The path of asr model in docker
PARAMS_DOCKER_ASR_PATH=""
#  The asr model ID in ModelScope
PARAMS_ASR_ID="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"

PARAMS_HOST_PORT="10095"
PARAMS_DOCKER_PORT="10095"
PARAMS_DECODER_THREAD_NUM="32"
PARAMS_IO_THREAD_NUM="8"
PARAMS_SSL_FLAG=1


echo -e "#############################################################"
echo -e "#          ${RED}OS${PLAIN}: ${OSID} ${OSNUM} ${OSVER}"
echo -e "#      ${RED}Kernel${PLAIN}: $(uname -m) Linux $(uname -r)"
echo -e "#         ${RED}CPU${PLAIN}: $(grep 'model name' /proc/cpuinfo | uniq | awk -F : '{print $2}' | sed 's/^[ \t]*//g' | sed 's/ \+/ /g') "
echo -e "#     ${RED}CPU NUM${PLAIN}: ${CPUNUM}"
echo -e "#         ${RED}RAM${PLAIN}: $(cat /proc/meminfo | grep 'MemTotal' | awk -F : '{print $2}' | sed 's/^[ \t]*//g') "
echo -e "#"
echo -e "#     ${RED}Version${PLAIN}: ${scriptVersion} "
echo -e "# ${RED}Modify Date${PLAIN}: ${scriptDate}"
echo -e "#############################################################"
echo

# Initialization step
case "$1" in
    install|-i|--install)
        rootNess
        paramsFromDefault
        parseInput $@
        paramsConfigure
        result=$?
        result=`expr ${result} + 0`
        if [ $result -eq 0 ]; then
            showAllParams "install"
            installFunasrDocker
            dockerRun "install"
            result=$?

            try_count=1
            while true
            do
                stage=`expr ${result} + 0`
                if [ $try_count -ge 10 ]; then
                    break
                else
                    # 98: cannot find progress from Docker
                    if [ $stage -eq 98 ]; then
                        dockerStop
                        dockerRun "start"
                        result=$?
                        let try_count=try_count+1
                    else
                        break
                    fi
                fi
            done
        fi
        ;;
    start|-s|--start)
        rootNess
        paramsFromDefault
        showAllParams "only_show"
        dockerRun "start"
        result=$?

        try_count=1
        while true
        do
            stage=`expr ${result} + 0`
            if [ $try_count -ge 10 ]; then
                break
            else
                # 98: cannot find progress from Docker
                if [ $stage -eq 98 ]; then
                    dockerStop
                    dockerRun "start"
                    result=$?
                    let try_count=try_count+1
                else
                    break
                fi
            fi
        done
        ;;
    restart|-r|--restart)
        rootNess
        paramsFromDefault
        showAllParams "only_show"
        dockerStop
        dockerRun "start"
        result=$?
        
        try_count=1
        while true
        do
            stage=`expr ${result} + 0`
            if [ $try_count -ge 10 ]; then
                break
            else
                # 98: cannot find progress from Docker
                if [ $stage -eq 98 ]; then
                    dockerStop
                    dockerRun "start"
                    result=$?
                    let try_count=try_count+1
                else
                    break
                fi
            fi
        done
        ;;
    stop|-p|--stop)
        rootNess
        paramsFromDefault
        dockerStop
        ;;
    remove|-m|--remove)
        rootNess
        paramsFromDefault
        dockerStop
        dockerRemove
        rm -f ${DEFAULT_FUNASR_CONFIG_FILE}
        rm -f ${DEFAULT_FUNASR_SERVER_CONFIG_FILE}
        ;;
    update|-u|--update)
        rootNess
        paramsFromDefault

        if [ $# -eq 3 ]; then
            type=$2
            val=$3
            if [ "$type" = "--asr_model" ] || [ "$type" = "--vad_model" ] || [ "$type" = "--punc_model" ]; then
                modelChange $type $val
            elif [ "$type" = "--decode_thread_num" ] || [ "$type" = "--io_thread_num" ]; then
                threadNumChange $type $val
            elif [ "$type" = "--host_port" ] || [ "$type" = "--docker_port" ]; then
                portChange $type $val
            elif [ "$type" = "--workspace" ]; then
                relativePathToFullPath $val
                PARAMS_FUNASR_LOCAL_WORKSPACE=$full_path
                if [ ! -z "$PARAMS_FUNASR_LOCAL_WORKSPACE" ]; then
                    mkdir -p $PARAMS_FUNASR_LOCAL_WORKSPACE
                fi
            elif [ "$type" = "--ssl" ]; then
                switch=`expr ${val} + 0`
                if [ $switch -eq 0 ]; then
                    PARAMS_SSL_FLAG=0
                else
                    PARAMS_SSL_FLAG=1
                fi
            else
                displayHelp
            fi
        else
            displayHelp
        fi

        initParameters
        complementParameters
        showAllParams "install"
        dockerStop
        dockerRun "start"
        result=$?

        try_count=1
        while true
        do
            stage=`expr ${result} + 0`
            if [ $try_count -ge 10 ]; then
                break
            else
                # 98: cannot find progress from Docker
                # 60: DOCKER_ID is empty
                if [ $stage -eq 98 ] || [ $stage -eq 60 ]; then
                    dockerStop
                    dockerRun "start"
                    result=$?
                    let try_count=try_count+1
                else
                    break
                fi
            fi
        done
        ;;
    client|-c|--client)
        rootNess
        paramsFromDefault
        parseInput $@
        sampleClientRun
        ;;
    show|-o|--show)
        rootNess
        paramsFromDefault
        showAllParams "only_show"
        ;;
    *)
        displayHelp
        exit 0
        ;;
esac
