#!/usr/bin/env bash

scriptVersion="0.0.4"
scriptDate="20230701"


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
cur_dir=`pwd`


DEFAULT_DOCKER_OFFLINE_CPU_ZH_LISTS="https://raw.githubusercontent.com/alibaba-damo-academy/FunASR/main/funasr/runtime/docs/docker_offline_cpu_zh_lists"
DEFAULT_DOCKER_IMAGE_LISTS=$DEFAULT_DOCKER_OFFLINE_CPU_ZH_LISTS
DEFAULT_FUNASR_DOCKER_URL="registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr"
DEFAULT_FUNASR_RUNTIME_SDK_NAME="funasr-runtime-sdk"
DEFAULT_FUNASR_LOCAL_WORKSPACE="${cur_dir}/${DEFAULT_FUNASR_RUNTIME_SDK_NAME}"
DEFAULT_FUNASR_CONFIG_DIR="/var/funasr"
DEFAULT_FUNASR_CONFIG_FILE="${DEFAULT_FUNASR_CONFIG_DIR}/config"
DEFAULT_FUNASR_WORKSPACE_DIR="/workspace/models"
DEFAULT_DOCKER_PORT="10095"
DEFAULT_PROGRESS_FILENAME="progress.txt"
DEFAULT_SERVER_EXEC_NAME="funasr-wss-server"
DEFAULT_DOCKER_EXEC_DIR="/workspace/FunASR/funasr/runtime/websocket/build/bin"
DEFAULT_DOCKER_EXEC_PATH=${DEFAULT_DOCKER_EXEC_DIR}/${DEFAULT_SERVER_EXEC_NAME}
DEFAULT_SAMPLES_NAME="funasr_samples"
DEFAULT_SAMPLES_URL="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/${DEFAULT_SAMPLES_NAME}.tar.gz"


SAMPLE_CLIENTS=( \
"Python" \
"Linux_Cpp" \
)
DOCKER_IMAGES=()
PROGRESS_TXT="/var/funasr/progress.txt"
ASR_PERCENT_INT=0
VAD_PERCENT_INT=0
PUNC_PERCENT_INT=0
ASR_TITLE="Downloading"
ASR_PERCENT="0"
ASR_SPEED="0KB/s"
ASR_REVISION=""
VAD_TITLE="Downloading"
VAD_PERCENT="0"
VAD_SPEED="0KB/s"
VAD_REVISION=""
PUNC_TITLE="Downloading"
PUNC_PERCENT="0"
PUNC_SPEED="0KB/s"
PUNC_REVISION=""

ServerProgress(){
    status_flag="STATUS:"
    stage=0
    wait=0
    server_status=""

    while true
    do
        if [ -f "$PROGRESS_TXT" ]; then
            break
        else
            sleep 1
            let wait=wait+1
            if [ ${wait} -ge 6 ]; then
                break
            fi
        fi
    done

    if [ ! -f "$PROGRESS_TXT" ]; then
        echo -e "    ${RED}The note of progress does not exist.($PROGRESS_TXT) ${PLAIN}"
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
                ASR_TITLE=${line#*:}
                continue
            fi
            result=$(echo $line | grep "percent:")
            if [ "$result" != "" ]; then
                ASR_PERCENT=${line#*:}
                continue
            fi
            result=$(echo $line | grep "speed:")
            if [ "$result" != "" ]; then
                ASR_SPEED=${line#*:}
                continue
            fi
            result=$(echo $line | grep "revision:")
            if [ "$result" != "" ]; then
                ASR_REVISION=${line#*:}
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
                VAD_TITLE=${line#*:}
                continue
            fi
            result=$(echo $line | grep "percent:")
            if [ "$result" != "" ]; then
                VAD_PERCENT=${line#*:}
                continue
            fi
            result=$(echo $line | grep "speed:")
            if [ "$result" != "" ]; then
                VAD_SPEED=${line#*:}
                continue
            fi
            result=$(echo $line | grep "revision:")
            if [ "$result" != "" ]; then
                VAD_REVISION=${line#*:}
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
                PUNC_TITLE=${line#*:}
                continue
            fi
            result=$(echo $line | grep "percent:")
            if [ "$result" != "" ]; then
                PUNC_PERCENT=${line#*:}
                continue
            fi
            result=$(echo $line | grep "speed:")
            if [ "$result" != "" ]; then
                PUNC_SPEED=${line#*:}
                continue
            fi
            result=$(echo $line | grep "revision:")
            if [ "$result" != "" ]; then
                PUNC_REVISION=${line#*:}
                continue
            fi
        elif [ $stage -eq 99 ]; then
            echo -e "    ${RED}ERROR: $line${PLAIN}"
        fi
    done < $PROGRESS_TXT

    if [ $stage -ne 99 ]; then
        DrawProgress "ASR " $ASR_TITLE $ASR_PERCENT $ASR_SPEED $ASR_REVISION $ASR_PERCENT_INT
        ASR_PERCENT_INT=$?
        DrawProgress "VAD " $VAD_TITLE $VAD_PERCENT $VAD_SPEED $VAD_REVISION $VAD_PERCENT_INT
        VAD_PERCENT_INT=$?
        DrawProgress "PUNC" $PUNC_TITLE $PUNC_PERCENT $PUNC_SPEED $PUNC_REVISION $PUNC_PERCENT_INT
        PUNC_PERCENT_INT=$?
    fi

    return $stage
}

DrawProgress(){
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

    LOADING_FLAG="Loading"
    if [ "$title" = "$LOADING_FLAG" ]; then
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
        printf "\r    \e[0;$color;1m[%s][%-11s][%-50s][%d%%][%s]\e[0m" "$model" "$title" "$str" "$$index" "$revision"
    else
        printf "\r    \e[0;$color;1m[%s][%-11s][%-50s][%3d%%][%8s][%s]\e[0m" "$model" "$title" "$str" "$index" "$speed" "$revision"
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

initConfiguration(){
    mkdir -p $DEFAULT_FUNASR_CONFIG_DIR
    if [ ! -f $DEFAULT_FUNASR_CONFIG_FILE ]; then
        touch $DEFAULT_FUNASR_CONFIG_FILE
    fi
}

initParameters(){
    # Init workspace in local by new parameters.
    PARAMS_FUNASR_SAMPLES_LOCAL_DIR=${PARAMS_FUNASR_LOCAL_WORKSPACE}/${DEFAULT_SAMPLES_NAME}
    PARAMS_FUNASR_LOCAL_MODELS_DIR="${PARAMS_FUNASR_LOCAL_WORKSPACE}/models"

    mkdir -p $PARAMS_FUNASR_LOCAL_WORKSPACE
    mkdir -p $PARAMS_FUNASR_LOCAL_MODELS_DIR
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
    list_url=$DEFAULT_DOCKER_IMAGE_LISTS
    while true
    do
        content=$(curl --connect-timeout 10 -m 10 -s $list_url)
        if [ ! -z "$content" ]; then
            break
        else
            echo -e "    ${RED}Unable to get docker image list due to network issues, try again.${PLAIN}"
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
    echo -e "${UNDERLINE}${BOLD}[0/6]${PLAIN}"
    echo -e "  ${YELLOW}Please check root access.${PLAIN}"
    echo

    echo -e "    ${WARNING} MUST RUN AS ${RED}ROOT${PLAIN} USER!"
    if [[ $EUID -ne 0 ]]; then
        echo -e "  ${ERROR} MUST RUN AS ${RED}ROOT${PLAIN} USER!"
    fi

    cd $cur_dir
    echo
}

# Set the host working path.
setupLocalWorkspaceDir(){
    echo -e "${UNDERLINE}${BOLD}[1/6]${PLAIN}"
    params_local_workspace=`sed '/^PARAMS_FUNASR_LOCAL_WORKSPACE=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    if [ -z "$params_local_workspace" ]; then
        params_local_workspace=$DEFAULT_FUNASR_LOCAL_WORKSPACE
    fi

    while true
    do
        echo -e "  ${YELLOW}Please enter the local path of workspace.${PLAIN}"
        echo -e "  Setting the local path of workspace, default(${CYAN}${params_local_workspace}${PLAIN}): \c"
        read PARAMS_FUNASR_LOCAL_WORKSPACE
        if [ -z "$PARAMS_FUNASR_LOCAL_WORKSPACE" ]; then
            if [ -z "$params_local_workspace" ]; then
                continue
            else
                PARAMS_FUNASR_LOCAL_WORKSPACE=$params_local_workspace
                mkdir -p $PARAMS_FUNASR_LOCAL_WORKSPACE
            fi
        fi
        if [ ! -d "$PARAMS_FUNASR_LOCAL_WORKSPACE" ]; then
            echo -e "    ${RED}The local workspace(${PARAMS_FUNASR_LOCAL_WORKSPACE}) set does not exist, please setup again.${PLAIN}"
        else
            PARAMS_FUNASR_LOCAL_MODELS_DIR="${PARAMS_FUNASR_LOCAL_WORKSPACE}/models"
            echo -e "  The local workspace path is ${GREEN}${PARAMS_FUNASR_LOCAL_WORKSPACE}${PLAIN} ."
            echo -e "  The models will store in local path(${GREEN}${PARAMS_FUNASR_LOCAL_MODELS_DIR}${PLAIN}) during the run."

            break
        fi
    done
    echo
}

# Get a list of docker images and select them.
selectDockerImages(){
    echo -e "${UNDERLINE}${BOLD}[2/6]${PLAIN}"
    echo -e "  ${YELLOW}Getting the list of docker images, please wait a few seconds.${PLAIN}"
    readDockerInfoFromUrl
    echo

    echo -e "  ${YELLOW}Please choose the Docker image.${PLAIN}"
    menuSelection ${DOCKER_IMAGES[*]}
    result=$?
    index=`expr $result - 1`

    PARAMS_DOCKER_IMAGE=${DOCKER_IMAGES[${index}]}
    echo -e "  ${UNDERLINE}You have chosen the Docker image:${PLAIN} ${GREEN}${PARAMS_DOCKER_IMAGE}${PLAIN}"

    checkDockerExist
    result=$?
    result=`expr $result + 0`
    if [ ${result} -eq 50 ]; then
        return 50
    fi

    echo
}

# Configure FunASR server host port setting.
setupHostPort(){
    echo -e "${UNDERLINE}${BOLD}[3/6]${PLAIN}"

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

    echo -e "${UNDERLINE}${BOLD}[1-3/6]${PLAIN}"
    echo -e "  ${YELLOW}Load parameters from ${DEFAULT_FUNASR_CONFIG_FILE}${PLAIN}"
    echo

    PARAMS_FUNASR_LOCAL_WORKSPACE=`sed '/^PARAMS_FUNASR_LOCAL_WORKSPACE=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_FUNASR_SAMPLES_LOCAL_DIR=`sed '/^PARAMS_FUNASR_SAMPLES_LOCAL_DIR=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_FUNASR_SAMPLES_LOCAL_PATH=`sed '/^PARAMS_FUNASR_SAMPLES_LOCAL_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_FUNASR_LOCAL_MODELS_DIR=`sed '/^PARAMS_FUNASR_LOCAL_MODELS_DIR=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_FUNASR_CONFIG_PATH=`sed '/^PARAMS_FUNASR_CONFIG_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`

    PARAMS_DOCKER_IMAGE=`sed '/^PARAMS_DOCKER_IMAGE=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_DOWNLOAD_MODEL_DIR=`sed '/^PARAMS_DOWNLOAD_MODEL_DIR=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_LOCAL_ASR_PATH=`sed '/^PARAMS_LOCAL_ASR_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_DOCKER_ASR_PATH=`sed '/^PARAMS_DOCKER_ASR_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_ASR_ID=`sed '/^PARAMS_ASR_ID=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_LOCAL_VAD_PATH=`sed '/^PARAMS_LOCAL_VAD_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_DOCKER_VAD_PATH=`sed '/^PARAMS_DOCKER_VAD_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_VAD_ID=`sed '/^PARAMS_VAD_ID=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_LOCAL_PUNC_PATH=`sed '/^PARAMS_LOCAL_PUNC_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_DOCKER_PUNC_PATH=`sed '/^PARAMS_DOCKER_PUNC_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_PUNC_ID=`sed '/^PARAMS_PUNC_ID=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_DOCKER_EXEC_PATH=`sed '/^PARAMS_DOCKER_EXEC_PATH=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_HOST_PORT=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_DOCKER_PORT=`sed '/^PARAMS_DOCKER_PORT=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_DECODER_THREAD_NUM=`sed '/^PARAMS_DECODER_THREAD_NUM=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
    PARAMS_IO_THREAD_NUM=`sed '/^PARAMS_IO_THREAD_NUM=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
}

saveParams(){
    echo "$i" > $DEFAULT_FUNASR_CONFIG_FILE
    echo -e "  ${GREEN}Parameters are stored in the file ${DEFAULT_FUNASR_CONFIG_FILE}${PLAIN}"

    echo "PARAMS_DOCKER_IMAGE=${PARAMS_DOCKER_IMAGE}" > $DEFAULT_FUNASR_CONFIG_FILE

    echo "PARAMS_FUNASR_LOCAL_WORKSPACE=${PARAMS_FUNASR_LOCAL_WORKSPACE}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_FUNASR_SAMPLES_LOCAL_DIR=${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_FUNASR_SAMPLES_LOCAL_PATH=${PARAMS_FUNASR_SAMPLES_LOCAL_PATH}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_FUNASR_LOCAL_MODELS_DIR=${PARAMS_FUNASR_LOCAL_MODELS_DIR}" >> $DEFAULT_FUNASR_CONFIG_FILE
    echo "PARAMS_FUNASR_CONFIG_PATH=${PARAMS_FUNASR_CONFIG_PATH}" >> $DEFAULT_FUNASR_CONFIG_FILE

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
}

showAllParams(){
    echo -e "${UNDERLINE}${BOLD}[4/6]${PLAIN}"
    echo -e "  ${YELLOW}Show parameters of FunASR server setting and confirm to run ...${PLAIN}"
    echo

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

    echo
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
    echo -e "${UNDERLINE}${BOLD}[5/6]${PLAIN}"

    if [ $DOCKERINFOLEN -gt 30 ]; then
        echo -e "  ${YELLOW}Docker has installed.${PLAIN}"
    else
        lowercase_osid=$(echo $OSID | tr '[A-Z]' '[a-z]')
        echo -e "  ${YELLOW}Start install docker for ${lowercase_osid} ${PLAIN}"
        DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun"
        DOCKER_INSTALL_RUN_CMD=""

        case "$lowercase_osid" in
            ubuntu)
                DOCKER_INSTALL_CMD="curl -fsSL https://test.docker.com -o test-docker.sh"
                DOCKER_INSTALL_RUN_CMD="sudo sh test-docker.sh"
                ;;
            centos)
                DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun"
                ;;
            debian)
                DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com -o get-docker.sh"
                DOCKER_INSTALL_RUN_CMD="sudo sh get-docker.sh"
                ;;
            \"alios\")
                DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com -o get-docker.sh"
                DOCKER_INSTALL_RUN_CMD="sudo sh get-docker.sh"
                ;;
            \"alinux\")
                DOCKER_INSTALL_CMD="sudo yum -y install dnf"
                DOCKER_INSTALL_RUN_CMD="sudo dnf -y install docker"
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
        sudo systemctl start docker

        DOCKERINFO=$(sudo docker info | wc -l)
        DOCKERINFOLEN=`expr ${DOCKERINFO} + 0`
        if [ $DOCKERINFOLEN -gt 30 ]; then
            echo -e "  ${GREEN}Docker install success, start docker server.${PLAIN}"
            sudo systemctl start docker
        else
            echo -e "  ${RED}Docker install failed!${PLAIN}"
            exit 1
        fi
    fi

    echo
    sleep 1

    # Download docker image
    echo -e "  ${YELLOW}Pull docker image(${PARAMS_DOCKER_IMAGE})...${PLAIN}"

    sudo docker pull $PARAMS_DOCKER_IMAGE

    echo
    sleep 1
}

dockerRun(){
    echo -e "${UNDERLINE}${BOLD}[6/6]${PLAIN}"
    echo -e "  ${YELLOW}Construct command and run docker ...${PLAIN}"

    RUN_CMD="sudo docker run"
    PORT_MAP=" -p ${PARAMS_HOST_PORT}:${PARAMS_DOCKER_PORT}"
    DIR_PARAMS=" --privileged=true"
    DIR_MAP_PARAMS=""
    if [ ! -z "$PARAMS_LOCAL_ASR_DIR" ]; then
        if [ -z "$DIR_MAP_PARAMS" ]; then
            DIR_MAP_PARAMS="${DIR_PARAMS} -v ${PARAMS_LOCAL_ASR_DIR}:${PARAMS_DOCKER_ASR_DIR}"
        else
            DIR_MAP_PARAMS="${DIR_MAP_PARAMS} -v ${PARAMS_LOCAL_ASR_DIR}:${PARAMS_DOCKER_ASR_DIR}"
        fi
    fi
    if [ ! -z "$PARAMS_LOCAL_VAD_DIR" ]; then
        if [ -z "$DIR_MAP_PARAMS" ]; then
            DIR_MAP_PARAMS="${DIR_PARAMS} -v ${PARAMS_LOCAL_VAD_DIR}:${PARAMS_DOCKER_VAD_DIR}"
        else
            DIR_MAP_PARAMS="${DIR_MAP_PARAMS} -v ${PARAMS_LOCAL_VAD_DIR}:${PARAMS_DOCKER_VAD_DIR}"
        fi
    fi
    if [ ! -z "$PARAMS_LOCAL_PUNC_DIR" ]; then
        if [ -z "$DIR_MAP_PARAMS" ]; then
            DIR_MAP_PARAMS="${DIR_PARAMS} -v ${PARAMS_LOCAL_PUNC_DIR}:${PARAMS_DOCKER_PUNC_DIR}"
        else
            DIR_MAP_PARAMS="${DIR_MAP_PARAMS} -v ${PARAMS_LOCAL_VAD_DIR}:${PARAMS_DOCKER_VAD_DIR}"
        fi
    fi

    EXEC_PARAMS="\"exec\":\"${PARAMS_DOCKER_EXEC_PATH}\""
    if [ ! -z "$PARAMS_ASR_ID" ]; then
        ASR_PARAMS="\"--model-dir\":\"${PARAMS_ASR_ID}\""
    else
        ASR_PARAMS="\"--model-dir\":\"${PARAMS_DOCKER_ASR_PATH}\""
    fi
    if [ ! -z "$PARAMS_VAD_ID" ]; then
        VAD_PARAMS="\"--vad-dir\":\"${PARAMS_VAD_ID}\""
    else
        VAD_PARAMS="\"--vad-dir\":\"${PARAMS_DOCKER_VAD_PATH}\""
    fi
    if [ ! -z "$PARAMS_PUNC_ID" ]; then
        PUNC_PARAMS="\"--punc-dir\":\"${PARAMS_PUNC_ID}\""
    else
        PUNC_PARAMS="\"--punc-dir\":\"${PARAMS_DOCKER_PUNC_PATH}\""
    fi
    DOWNLOAD_PARARMS="\"--download-model-dir\":\"${PARAMS_DOWNLOAD_MODEL_DIR}\""
    if [ -z "$PARAMS_DOWNLOAD_MODEL_DIR" ]; then
        MODEL_PARAMS="${ASR_PARAMS},${VAD_PARAMS},${PUNC_PARAMS}"
    else
        MODEL_PARAMS="${ASR_PARAMS},${VAD_PARAMS},${PUNC_PARAMS},${DOWNLOAD_PARARMS}"
    fi

    DECODER_PARAMS="\"--decoder-thread-num\":\"${PARAMS_DECODER_THREAD_NUM}\""
    IO_PARAMS="\"--io-thread-num\":\"${PARAMS_IO_THREAD_NUM}\""
    THREAD_PARAMS=${DECODER_PARAMS},${IO_PARAMS}
    PORT_PARAMS="\"--port\":\"${PARAMS_DOCKER_PORT}\""
    CRT_PATH="\"--certfile\":\"/workspace/FunASR/funasr/runtime/ssl_key/server.crt\""
    KEY_PATH="\"--keyfile\":\"/workspace/FunASR/funasr/runtime/ssl_key/server.key\""

    ENV_PARAMS=" -v /var/funasr:/workspace/.config"
    ENV_PARAMS=" ${ENV_PARAMS} --env DAEMON_SERVER_CONFIG={\"server\":[{${EXEC_PARAMS},${MODEL_PARAMS},${THREAD_PARAMS},${PORT_PARAMS},${CRT_PATH},${KEY_PATH}}]}"

    RUN_CMD="${RUN_CMD}${PORT_MAP}${DIR_MAP_PARAMS}${ENV_PARAMS}"
    RUN_CMD="${RUN_CMD} -it -d ${PARAMS_DOCKER_IMAGE}"

    # check Docker
    checkDockerExist
    result=$?
    result=`expr ${result} + 0`
    if [ ${result} -eq 50 ]; then
        return 50
    fi

    server_log="/var/funasr/server_console.log"
    rm -f ${PROGRESS_TXT}
    rm -f ${server_log}

    ${RUN_CMD}

    echo
    echo -e "  ${YELLOW}Loading models:${PLAIN}"

    # Hide the cursor, start draw progress.
    printf "\e[?25l"
    while true
    do
        ServerProgress
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

    downloadSamples
    echo -e "  ${BOLD}The sample code is already stored in the ${PLAIN}(${GREEN}${PARAMS_FUNASR_SAMPLES_LOCAL_DIR}${PLAIN}) ."
    echo -e "  ${BOLD}If you want to see an example of how to use the client, you can run ${PLAIN}${GREEN}sudo bash funasr-runtime-deploy-offline-cpu-zh.sh -c${PLAIN} ."
    echo
}

downloadSamples(){
    if [ ! -d $PARAMS_FUNASR_SAMPLES_LOCAL_DIR ]; then
        echo -e "${YELLOW}Downloading samples to $PARAMS_FUNASR_LOCAL_WORKSPACE ...${PLAIN}"

        DOWNLOAD_CMD="curl ${DEFAULT_SAMPLES_URL} -o ${PARAMS_FUNASR_SAMPLES_LOCAL_PATH}"
        UNTAR_CMD="tar -zxf ${PARAMS_FUNASR_SAMPLES_LOCAL_PATH} -C ${PARAMS_FUNASR_LOCAL_WORKSPACE}"

        if [ ! -f "$PARAMS_FUNASR_SAMPLES_LOCAL_PATH" ]; then
            $DOWNLOAD_CMD
        fi
        $UNTAR_CMD
    fi
}

checkDockerExist(){
    result=$(sudo docker ps | grep ${PARAMS_DOCKER_IMAGE} | wc -l)
    result=`expr ${result} + 0`
    if [ ${result} -ne 0 ]; then
        echo
        echo -e "  ${RED}Docker: ${PARAMS_DOCKER_IMAGE} has been launched, please run (${PLAIN}${GREEN}sudo bash funasr-runtime-deploy.sh -p${PLAIN}${RED}) to stop Docker first.${PLAIN}"
        return 50
    fi
}

dockerExit(){
    echo -e "  ${YELLOW}Stop docker(${PARAMS_DOCKER_IMAGE}) server ...${PLAIN}"
    sudo docker stop `sudo docker ps -a| grep ${PARAMS_DOCKER_IMAGE} | awk '{print $1}' `
    echo
    sleep 1
}

modelChange(){
    model_type=$1
    model_id=$2
    local_flag=0

    if [ -d "$model_id" ]; then
        local_flag=1
    else
        local_flag=0
    fi

    result=$(echo $model_type | grep "asr_model")
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
                middle=${PARAMS_LOCAL_ASR_DIR#*"${PARAMS_FUNASR_LOCAL_MODELS_DIR}"}
                PARAMS_DOCKER_ASR_DIR=$PARAMS_DOWNLOAD_MODEL_DIR
                PARAMS_DOCKER_ASR_PATH=${PARAMS_DOCKER_ASR_DIR}/${middle}/${model_name}
            fi
        fi
    fi
    result=$(echo ${model_type} | grep "vad_model")
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
                middle=${PARAMS_LOCAL_VAD_DIR#*"${PARAMS_FUNASR_LOCAL_MODELS_DIR}"}
                PARAMS_DOCKER_VAD_DIR=$PARAMS_DOWNLOAD_MODEL_DIR
                PARAMS_DOCKER_VAD_PATH=${PARAMS_DOCKER_VAD_DIR}/${middle}/${model_name}
            fi
        fi
    fi
    result=$(echo $model_type | grep "punc_model")
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
            middle=${PARAMS_LOCAL_PUNC_DIR#*"${PARAMS_FUNASR_LOCAL_MODELS_DIR}"}
            PARAMS_DOCKER_PUNC_DIR=$PARAMS_DOWNLOAD_MODEL_DIR
            PARAMS_DOCKER_PUNC_PATH=${PARAMS_DOCKER_PUNC_DIR}/${middle}/${model_name}
        fi
    fi
}

threadNumChange() {
    type=$1
    val=$2

    if [ -z "$val"]; then
        num=`expr ${val} + 0`
        if [ $num -ge 1 ] && [ $num -le 1024 ]; then
            result=$(echo ${type} | grep "decode_thread_num")
            if [ "$result" != "" ]; then
                PARAMS_DECODER_THREAD_NUM=$num
            fi
            result=$(echo ${type} | grep "io_thread_num")
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

    DOWNLOAD_CMD="curl ${DEFAULT_SAMPLES_URL} -o ${PARAMS_FUNASR_SAMPLES_LOCAL_PATH}"
    UNTAR_CMD="tar -zxf ${PARAMS_FUNASR_SAMPLES_LOCAL_PATH} ${PARAMS_FUNASR_LOCAL_WORKSPACE}"

    if [ ! -f "$PARAMS_FUNASR_SAMPLES_LOCAL_PATH" ]; then
        $DOWNLOAD_CMD
    fi    
    if [ -f "$PARAMS_FUNASR_SAMPLES_LOCAL_PATH" ]; then
        $UNTAR_CMD
    fi
    if [ -d "$PARAMS_FUNASR_SAMPLES_LOCAL_DIR" ]; then

        echo -e "  Please select the client you want to run."
        menuSelection ${SAMPLE_CLIENTS[*]}
        result=$?
        index=`expr ${result} - 1`
        lang=${SAMPLE_CLIENTS[${index}]}
        echo

        SERVER_IP="127.0.0.1"
        read -p "  Please enter the IP of server, default($SERVER_IP): " SERVER_IP
        if [ -z "$SERVER_IP" ]; then
            SERVER_IP="127.0.0.1"
        fi

        HOST_PORT=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
        if [ -z "$HOST_PORT" ]; then
            HOST_PORT="10095"
        fi
        read -p "  Please enter the port of server, default($HOST_PORT): " HOST_PORT
        if [ -z "$HOST_PORT" ]; then
            HOST_PORT=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${DEFAULT_FUNASR_CONFIG_FILE}`
            if [ -z "$HOST_PORT" ]; then
                HOST_PORT="10095"
            fi
        fi

        WAV_PATH="${PARAMS_FUNASR_LOCAL_WORKSPACE}/funasr_samples/audio/asr_example.wav"
        read -p "  Please enter the audio path, default(${WAV_PATH}): " WAV_PATH
        if [ -z "$WAV_PATH" ]; then
            WAV_PATH="${PARAMS_FUNASR_LOCAL_WORKSPACE}/funasr_samples/audio/asr_example.wav"
        fi

        echo
        PRE_CMD=”“
        case "$lang" in
            Linux_Cpp)
                PRE_CMD="export LD_LIBRARY_PATH=${PARAMS_FUNASR_LOCAL_WORKSPACE}/funasr_samples/cpp/libs:\$LD_LIBRARY_PATH"
                CLIENT_EXEC="${PARAMS_FUNASR_LOCAL_WORKSPACE}/funasr_samples/cpp/funasr-wss-client"
                RUN_CMD="${CLIENT_EXEC} --server-ip ${SERVER_IP} --port ${HOST_PORT} --wav-path ${WAV_PATH}"
                echo -e "  Run ${BLUE}${PRE_CMD}${PLAIN}"
                $PRE_CMD
                echo
                ;;
            Python)
                CLIENT_EXEC="${PARAMS_FUNASR_LOCAL_WORKSPACE}/funasr_samples/python/wss_client_asr.py"
                RUN_CMD="python3 ${CLIENT_EXEC} --host ${SERVER_IP} --port ${HOST_PORT} --mode offline --audio_in ${WAV_PATH} --send_without_sleep --output_dir ${PARAMS_FUNASR_LOCAL_WORKSPACE}/funasr_samples/python"
                PRE_CMD="pip3 install click>=8.0.4"
                echo -e "  Run ${BLUE}${PRE_CMD}${PLAIN}"
                $PRE_CMD
                echo
                PRE_CMD="pip3 install -r ${PARAMS_FUNASR_LOCAL_WORKSPACE}/funasr_samples/python/requirements_client.txt"
                echo -e "  Run ${BLUE}${PRE_CMD}${PLAIN}"
                $PRE_CMD
                echo
                ;;
            *)
                echo "${lang} is not supported."
                ;;
        esac

        echo -e "  Run ${BLUE}${RUN_CMD}${PLAIN}"
        $RUN_CMD
        echo
        echo -e "  If failed, you can try (${GREEN}${RUN_CMD}${PLAIN}) in your Shell."
        echo
    fi
}

paramsConfigure(){
    initConfiguration
    setupLocalWorkspaceDir
    initParameters
    selectDockerImages
    result=$?
    result=`expr ${result} + 0`
    if [ ${result} -eq 50 ]; then
        return 50
    fi

    setupHostPort
    complementParameters
}

# Display Help info
displayHelp(){
    echo -e "${UNDERLINE}Usage${PLAIN}:"
    echo -e "  $0 [OPTIONAL FLAGS]"
    echo
    echo -e "funasr-runtime-deploy-offline-cpu.sh - a Bash script to install&run FunASR docker."
    echo
    echo -e "${UNDERLINE}Options${PLAIN}:"
    echo -e "   ${BOLD}-i, install, --install${PLAIN}    Install and run FunASR docker."
    echo -e "   ${BOLD}-s, start  , --start${PLAIN}      Run FunASR docker with configuration that has already been set."
    echo -e "   ${BOLD}-p, stop   , --stop${PLAIN}       Stop FunASR docker."
    echo -e "   ${BOLD}-r, restart, --restart${PLAIN}    Restart FunASR docker."
    echo -e "   ${BOLD}-u, update , --update${PLAIN}     Update parameters that has already been set."
    echo -e "                --update [asr_model | vad_model | punc_model] <model_id or local model path>"
    echo -e "                --update [host_port | docker_port] <port number>"
    echo -e "                --update [decode_thread_num | io_thread_num] <the number of threads>"
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
                    DEFAULT_FUNASR_LOCAL_WORKSPACE=$val
                fi
            fi
        done
    fi
}

# OS
OSID=$(grep ^ID= /etc/os-release | cut -d= -f2)
OSVER=$(lsb_release -cs)
OSNUM=$(grep -oE  "[0-9.]+" /etc/issue)
CPUNUM=$(cat /proc/cpuinfo |grep "processor"|wc -l)
DOCKERINFO=$(sudo docker info | wc -l)
DOCKERINFOLEN=`expr ${DOCKERINFO} + 0`

# PARAMS
#  The workspace for FunASR in local
PARAMS_FUNASR_LOCAL_WORKSPACE=$DEFAULT_FUNASR_LOCAL_WORKSPACE
#  The dir stored sample code in local
PARAMS_FUNASR_SAMPLES_LOCAL_DIR=${PARAMS_FUNASR_LOCAL_WORKSPACE}/${DEFAULT_SAMPLES_NAME}
#  The path of sample code in local
PARAMS_FUNASR_SAMPLES_LOCAL_PATH=${PARAMS_FUNASR_LOCAL_WORKSPACE}/${DEFAULT_SAMPLES_NAME}.tar.gz
#  The dir stored models in local
PARAMS_FUNASR_LOCAL_MODELS_DIR="${PARAMS_FUNASR_LOCAL_WORKSPACE}/models"
#  The path of configuration in local
PARAMS_FUNASR_CONFIG_PATH="${PARAMS_FUNASR_LOCAL_WORKSPACE}/config"

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
        parseInput $@
        paramsConfigure
        result=$?
        result=`expr ${result} + 0`
        if [ ${result} -ne 50 ]; then
            showAllParams
            installFunasrDocker
            dockerRun
            result=$?
            stage=`expr ${result} + 0`
            if [ $stage -eq 98 ]; then
                dockerExit
                dockerRun
            fi
        fi
        ;;
    start|-s|--start)
        rootNess
        paramsFromDefault
        showAllParams
        dockerRun
        result=$?
        stage=`expr ${result} + 0`
        if [ $stage -eq 98 ]; then
            dockerExit
            dockerRun
        fi
        ;;
    restart|-r|--restart)
        rootNess
        paramsFromDefault
        showAllParams
        dockerExit
        dockerRun
        result=$?
        stage=`expr ${result} + 0`
        if [ $stage -eq 98 ]; then
            dockerExit
            dockerRun
        fi
        ;;
    stop|-p|--stop)
        rootNess
        paramsFromDefault
        dockerExit
        ;;
    update|-u|--update)
        rootNess
        paramsFromDefault

        if [ $# -eq 3 ]; then
            type=$2
            val=$3
            if [ "$type" = "asr_model" ] || [ "$type" = "vad_model" ] || [ "$type" = "punc_model" ]; then
                modelChange $type $val
            elif [ "$type" = "decode_thread_num" ] || [ "$type" = "io_thread_num" ]; then
                threadNumChange $type $val
            elif [ "$type" = "host_port" ] || [ "$type" = "docker_port" ]; then
                portChange $type $val
            else
                setupLocalWorkspaceDir
            fi
        else
            setupLocalWorkspaceDir
        fi

        complementParameters
        initParameters
        showAllParams
        dockerExit
        dockerRun
        result=$?
        stage=`expr ${result} + 0`
        if [ $stage -eq 98 ]; then
            dockerExit
            dockerRun
        fi
        ;;
    client|-c|--client)
        rootNess
        paramsFromDefault
        sampleClientRun
        ;;
    show|-o|--show)
        rootNess
        paramsFromDefault
        showAllParams
        ;;
    *)
        displayHelp
        exit 0
        ;;
esac
