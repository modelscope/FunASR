#!/usr/bin/env bash

scriptVersion="0.0.3"
scriptDate="20230629"

clear


# Set color
RED="\033[31;1m"
GREEN="\033[32;1m"
YELLOW="\033[33;1m"
BLUE="\033[34;1m"
CYAN="\033[36;1m"
PLAIN="\033[0m"

# Info messages
ERROR="${RED}[ERROR]${PLAIN}"
WARNING="${YELLOW}[WARNING]${PLAIN}"

# Font Format
BOLD="\033[1m"
UNDERLINE="\033[4m"

# Current folder
cur_dir=`pwd`


checkConfigFileAndTouch(){
    mkdir -p /var/funasr
    if [ ! -f $FUNASR_CONFIG_FILE ]; then
        touch $FUNASR_CONFIG_FILE
    fi
}

SAMPLE_CLIENTS=( \
"Python" \
"Linux_Cpp" \
)
ASR_MODELS=( \
"damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx" \
"model_name" \
"model_path" \
)
VAD_MODELS=( \
"damo/speech_fsmn_vad_zh-cn-16k-common-onnx" \
"model_name" \
"model_path" \
)
PUNC_MODELS=( \
"damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx" \
"model_name" \
"model_path" \
)
DOCKER_IMAGES=( \
"registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest" \
"registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0" \
)
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
        if [ ${menu_no} -ge ${len} ]; then
            break
        fi
    done

    while true
    do
        read -p "  Enter your choice: " result

        expr ${result} + 0 &>/dev/null
        if [ $? -eq 0 ]; then
            if [ ${result} -ge 1 ] && [ ${result} -le ${len} ]; then
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
            if [ ${wait} -ge 10 ]; then
                break
            fi
        fi
    done

    if [ ! -f "$PROGRESS_TXT" ]; then
        echo -e "    ${RED}The note of progress does not exist.(${PROGRESS_TXT}) ${PLAIN}"
        return 98
    fi

    stage=1
    while read line
    do
        if [ $stage -eq 1 ]; then
            result=$(echo $line | grep "STATUS:")
            if [[ "$result" != "" ]]
            then
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
            if [[ "$result" != "" ]]
            then
                stage=3
                continue
            fi
        elif [ $stage -eq 3 ]; then
            result=$(echo $line | grep "VAD")
            if [[ "$result" != "" ]]
            then
                stage=4
                continue
            fi
            result=$(echo $line | grep "title:")
            if [[ "$result" != "" ]]
            then
                ASR_TITLE=${line#*:}
                continue
            fi
            result=$(echo $line | grep "percent:")
            if [[ "$result" != "" ]]
            then
                ASR_PERCENT=${line#*:}
                continue
            fi
            result=$(echo $line | grep "speed:")
            if [[ "$result" != "" ]]
            then
                ASR_SPEED=${line#*:}
                continue
            fi
            result=$(echo $line | grep "revision:")
            if [[ "$result" != "" ]]
            then
                ASR_REVISION=${line#*:}
                continue
            fi
        elif [ $stage -eq 4 ]; then
            result=$(echo $line | grep "PUNC")
            if [[ "$result" != "" ]]
            then
                stage=5
                continue
            fi
            result=$(echo $line | grep "title:")
            if [[ "$result" != "" ]]
            then
                VAD_TITLE=${line#*:}
                continue
            fi
            result=$(echo $line | grep "percent:")
            if [[ "$result" != "" ]]
            then
                VAD_PERCENT=${line#*:}
                continue
            fi
            result=$(echo $line | grep "speed:")
            if [[ "$result" != "" ]]
            then
                VAD_SPEED=${line#*:}
                continue
            fi
            result=$(echo $line | grep "revision:")
            if [[ "$result" != "" ]]
            then
                VAD_REVISION=${line#*:}
                continue
            fi
        elif [ $stage -eq 5 ]; then
            result=$(echo $line | grep "DONE")
            if [[ "$result" != "" ]]
            then
                # Done and break.
                stage=6
                break
            fi
            result=$(echo $line | grep "title:")
            if [[ "$result" != "" ]]
            then
                PUNC_TITLE=${line#*:}
                continue
            fi
            result=$(echo $line | grep "percent:")
            if [[ "$result" != "" ]]
            then
                PUNC_PERCENT=${line#*:}
                continue
            fi
            result=$(echo $line | grep "speed:")
            if [[ "$result" != "" ]]
            then
                PUNC_SPEED=${line#*:}
                continue
            fi
            result=$(echo $line | grep "revision:")
            if [[ "$result" != "" ]]
            then
                PUNC_REVISION=${line#*:}
                continue
            fi
        elif [ $stage -eq 99 ]; then
            echo -e "    ${RED}ERROR: $line${PLAIN}"
        fi
    done  < $PROGRESS_TXT

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

# Make sure root user
rootNess(){
    echo -e "${UNDERLINE}${BOLD}[0/9]${PLAIN}"
    echo -e "  ${YELLOW}Please check root access.${PLAIN}"
    echo

    echo -e "  ${WARNING} MUST RUN AS ${RED}ROOT${PLAIN} USER!"
    if [[ $EUID -ne 0 ]]; then
        echo -e "  ${ERROR} MUST RUN AS ${RED}ROOT${PLAIN} USER!"
    fi

    checkConfigFileAndTouch
    cd ${cur_dir}
    echo
}

selectDockerImages(){
    echo -e "${UNDERLINE}${BOLD}[1/9]${PLAIN}"
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

setupModelType(){
    echo -e "${UNDERLINE}${BOLD}[2/9]${PLAIN}"
    echo -e "  ${YELLOW}Please input [Y/n] to confirm whether to automatically download model_id in ModelScope or use a local model.${PLAIN}"
    echo -e "  [y] With the model in ModelScope, the model will be automatically downloaded to Docker(${CYAN}/workspace/models${PLAIN})."
    echo -e "      If you select both the local model and the model in ModelScope, select [y]."
    echo "  [n] Use the models on the localhost, the directory where the model is located will be mapped to Docker."

    while true
    do
        read -p "  Setting confirmation[Y/n]: " model_id_flag

        if [ -z "$model_id_flag" ]; then
            model_id_flag="y"
        fi
        YES="Y"
        yes="y"
        NO="N"
        no="n"
        if [ "$model_id_flag" = "$YES" ] || [ "$model_id_flag" = "$yes" ]; then
            # please set model_id later.
            PARAMS_DOWNLOAD_MODEL_DIR="/workspace/models"
            echo -e "  ${UNDERLINE}You have chosen to use the model in ModelScope, please set the model ID in the next steps, and the model will be automatically downloaded in (${PARAMS_DOWNLOAD_MODEL_DIR}) during the run.${PLAIN}"

            params_local_models_dir=`sed '/^PARAMS_LOCAL_MODELS_DIR=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
            if [ -z "$params_local_models_dir" ]; then
                params_local_models_dir="${cur_dir}/models"
                mkdir -p ${params_local_models_dir}
            fi
            while true
            do
                echo
                echo -e "  ${YELLOW}Please enter the local path to download models, the corresponding path in Docker is ${PARAMS_DOWNLOAD_MODEL_DIR}.${PLAIN}"
                read -p "  Setting the local path to download models, default(${params_local_models_dir}): " PARAMS_LOCAL_MODELS_DIR
                if [ -z "$PARAMS_LOCAL_MODELS_DIR" ]; then
                    if [ -z "$params_local_models_dir" ]; then
                        echo -e "    ${RED}The local path set is empty, please setup again.${PLAIN}"
                        continue
                    else
                        PARAMS_LOCAL_MODELS_DIR=$params_local_models_dir
                    fi
                fi
                if [ ! -d "$PARAMS_LOCAL_MODELS_DIR" ]; then
                    echo -e "    ${RED}The local model path(${PARAMS_LOCAL_MODELS_DIR}) set does not exist, please setup again.${PLAIN}"
                else
                    echo -e "  The local path(${GREEN}${PARAMS_LOCAL_MODELS_DIR}${PLAIN}) set will store models during the run."
                    break
                fi
            done

            break
        elif [ "$model_id_flag" = "$NO" ] || [ "$model_id_flag" = "$no" ]; then
            # download_model_dir is empty, will use models in localhost.
            PARAMS_DOWNLOAD_MODEL_DIR=""
            PARAMS_LOCAL_MODELS_DIR=""
            echo -e "  ${UNDERLINE}You have chosen to use models from the localhost, set the path to each model in the localhost in the next steps.${PLAIN}"
            echo
            break
        fi
    done

    echo
}

# Set asr model for FunASR server
setupAsrModelId(){
    echo -e "  ${UNDERLINE}${BOLD}[2.1/9]${PLAIN}"

    if [ -z "$PARAMS_DOWNLOAD_MODEL_DIR" ]; then
        # download_model_dir is empty, will use models in localhost.
        params_local_asr_path=`sed '/^PARAMS_LOCAL_ASR_PATH=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
        if [ -z "$params_local_asr_path" ]; then
            PARAMS_LOCAL_ASR_PATH=""
        else
            PARAMS_LOCAL_ASR_PATH=${params_local_asr_path}
        fi

        echo -e "    ${YELLOW}Please input ASR model path in local for FunASR server.${PLAIN}"
        echo -e "    Default: ${CYAN}${PARAMS_LOCAL_ASR_PATH}${PLAIN}"

        while true
        do
            read -p "    Setting ASR model path in localhost: " PARAMS_LOCAL_ASR_PATH

            if [ -z "$PARAMS_LOCAL_ASR_PATH" ]; then
                PARAMS_LOCAL_ASR_PATH=${params_local_asr_path}
            fi
            if [ -z "$PARAMS_LOCAL_ASR_PATH" ]; then
                # use default asr model in Docker
                PARAMS_LOCAL_ASR_DIR=""
                PARAMS_DOCKER_ASR_DIR=""
                PARAMS_DOCKER_ASR_PATH="/workspace/models/asr"
                echo -e "    ${RED}Donnot set the local ASR model path, will use ASR model(${CYAN}/workspace/models/asr${PLAIN}${RED}) in Docker.${PLAIN}"

                echo -e "    ${UNDERLINE}You have chosen the default model dir in localhost:${PLAIN} ${GREEN}${PARAMS_LOCAL_ASR_PATH}${PLAIN}"
                echo -e "    ${UNDERLINE}The defalut model dir in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_ASR_PATH}${PLAIN}"
                break
            else
                if [ ! -d "$PARAMS_LOCAL_ASR_PATH" ]; then
                    echo -e "    ${RED}The ASR model path set does not exist, please setup again.${PLAIN}"
                else
                    # use asr model in localhost
                    PARAMS_LOCAL_ASR_DIR=$(dirname "$PARAMS_LOCAL_ASR_PATH")
                    asr_name=$(basename "$PARAMS_LOCAL_ASR_PATH")
                    PARAMS_DOCKER_ASR_DIR="/workspace/user_asr"
                    PARAMS_DOCKER_ASR_PATH=${PARAMS_DOCKER_ASR_DIR}/${asr_name}

                    echo -e "    ${UNDERLINE}You have chosen the model dir in localhost:${PLAIN} ${GREEN}${PARAMS_LOCAL_ASR_PATH}${PLAIN}"
                    echo -e "    ${UNDERLINE}The model dir in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_ASR_PATH}${PLAIN}"
                    break
                fi
            fi
        done

        PARAMS_ASR_ID=""
    else
        # please set model_id later.
        echo -e "    ${YELLOW}Please select ASR model_id in ModelScope from the list below.${PLAIN}"

        menuSelection ${ASR_MODELS[*]}
        result=$?
        index=`expr $result - 1`
        PARAMS_ASR_ID=${ASR_MODELS[${index}]}

        OTHERS="model_name"
        LOCAL_MODEL="model_path"
        if [ "$PARAMS_ASR_ID" = "$OTHERS" ]; then
            params_asr_id=`sed '/^PARAMS_ASR_ID=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
            if [ -z "$params_asr_id" ]; then
                PARAMS_ASR_ID=""
            else
                PARAMS_ASR_ID=${params_asr_id}
            fi

            echo -e "    Default: ${CYAN}${PARAMS_ASR_ID}${PLAIN}"

            while true
            do
                read -p "    Setting ASR model_id in ModelScope: " PARAMS_ASR_ID

                PARAMS_LOCAL_ASR_DIR=""
                PARAMS_LOCAL_ASR_PATH=""
                PARAMS_DOCKER_ASR_DIR=""
                if [ -z "$PARAMS_ASR_ID" ]; then
                    echo -e "    ${RED}The ASR model ID is empty, please setup again.${PLAIN}"
                else
                    break
                fi
            done
        elif [ "$PARAMS_ASR_ID" = "$LOCAL_MODEL" ]; then
            PARAMS_ASR_ID=""
            echo -e "    Please input ASR model path in local for FunASR server."

            while true
            do
                read -p "    Setting ASR model path in localhost: " PARAMS_LOCAL_ASR_PATH
                if [ -z "$PARAMS_LOCAL_ASR_PATH" ]; then
                    # use default asr model in Docker
                    echo -e "    ${RED}Please do not set an empty path in localhost.${PLAIN}"
                else
                    if [ ! -d "$PARAMS_LOCAL_ASR_PATH" ]; then
                        echo -e "    ${RED}The ASR model path set does not exist, please setup again.${PLAIN}"
                    else
                        # use asr model in localhost
                        PARAMS_LOCAL_ASR_DIR=$(dirname "$PARAMS_LOCAL_ASR_PATH")
                        asr_name=$(basename "$PARAMS_LOCAL_ASR_PATH")
                        PARAMS_DOCKER_ASR_DIR="/workspace/user_asr"
                        PARAMS_DOCKER_ASR_PATH=${PARAMS_DOCKER_ASR_DIR}/${asr_name}

                        echo -e "    ${UNDERLINE}You have chosen the model dir in localhost:${PLAIN} ${GREEN}${PARAMS_LOCAL_ASR_PATH}${PLAIN}"
                        echo -e "    ${UNDERLINE}The model dir in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_ASR_PATH}${PLAIN}"
                        echo
                        return 0
                    fi
                fi
            done
        fi

        PARAMS_DOCKER_ASR_DIR=$PARAMS_DOWNLOAD_MODEL_DIR
        PARAMS_DOCKER_ASR_PATH=${PARAMS_DOCKER_ASR_DIR}/${PARAMS_ASR_ID}

        echo -e "    ${UNDERLINE}The model ID is${PLAIN} ${GREEN}${PARAMS_ASR_ID}${PLAIN}"
        echo -e "    ${UNDERLINE}The model dir in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_ASR_PATH}${PLAIN}"
    fi

    echo
}

# Set vad model for FunASR server
setupVadModelId(){
    echo -e "  ${UNDERLINE}${BOLD}[2.2/9]${PLAIN}"

    if [ -z "$PARAMS_DOWNLOAD_MODEL_DIR" ]; then
        # download_model_dir is empty, will use models in localhost.
        params_local_vad_path=`sed '/^PARAMS_LOCAL_VAD_PATH=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
        if [ -z "$params_local_vad_path" ]; then
            PARAMS_LOCAL_VAD_PATH=""
        else
            PARAMS_LOCAL_VAD_PATH=${params_local_vad_path}
        fi

        echo -e "    ${YELLOW}Please input VAD model path in local for FunASR server.${PLAIN}"
        echo -e "    Default: ${CYAN}${PARAMS_LOCAL_VAD_PATH}${PLAIN}"

        while true
        do
            read -p "    Setting VAD model path in localhost: " PARAMS_LOCAL_VAD_PATH

            if [ -z "$PARAMS_LOCAL_VAD_PATH" ]; then
                PARAMS_LOCAL_VAD_PATH=${params_local_vad_path}
            fi
            if [ -z "$PARAMS_LOCAL_VAD_PATH" ]; then
                # use default vad model in Docker
                PARAMS_LOCAL_VAD_DIR=""
                PARAMS_DOCKER_VAD_DIR=""
                PARAMS_DOCKER_VAD_PATH="/workspace/models/vad"
                echo -e "    ${RED}Donnot set the local VAD model path, will use VAD model(${CYAN}/workspace/models/vad${PLAIN}${RED}) in Docker.${PLAIN}"

                echo -e "    ${UNDERLINE}You have chosen the default model dir in localhost${PLAIN}: ${GREEN}${PARAMS_LOCAL_VAD_PATH}${PLAIN}"
                echo -e "    ${UNDERLINE}The defalut model dir in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_VAD_PATH}${PLAIN}"
                break
            else
                if [ ! -d "$PARAMS_LOCAL_VAD_PATH" ]; then
                    echo -e "    ${RED}The VAD model path set does not exist, please setup again.${PLAIN}"
                else
                    # use vad model in localhost
                    PARAMS_LOCAL_VAD_DIR=$(dirname "$PARAMS_LOCAL_VAD_PATH")
                    vad_name=$(basename "$PARAMS_LOCAL_VAD_PATH")
                    PARAMS_DOCKER_VAD_DIR="/workspace/user_vad"
                    PARAMS_DOCKER_VAD_PATH=${PARAMS_DOCKER_VAD_DIR}/${vad_name}

                    echo -e "    ${UNDERLINE}You have chosen the model dir in localhost:${PLAIN} ${GREEN}${PARAMS_LOCAL_VAD_PATH}${PLAIN}"
                    echo -e "    ${UNDERLINE}The model dir in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_VAD_PATH}${PLAIN}"
                    break
                fi
            fi
        done

        PARAMS_VAD_ID=""
    else
        # please set model_id later.
        echo -e "    ${YELLOW}Please select VAD model_id in ModelScope from the list below.${PLAIN}"

        menuSelection ${VAD_MODELS[*]}
        result=$?
        index=`expr $result - 1`
        PARAMS_VAD_ID=${VAD_MODELS[${index}]}

        OTHERS="model_name"
        LOCAL_MODEL="model_path"
        if [ "$PARAMS_VAD_ID" = "$OTHERS" ]; then
            params_vad_id=`sed '/^PARAMS_VAD_ID=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
            if [ -z "$params_vad_id" ]; then
                PARAMS_VAD_ID=""
            else
                PARAMS_VAD_ID=${params_vad_id}
            fi

            echo -e "    Default: ${CYAN}${PARAMS_VAD_ID}${PLAIN}"

            while true
            do
                read -p "    Setting VAD model_id in ModelScope: " PARAMS_VAD_ID

                PARAMS_LOCAL_VAD_DIR=""
                PARAMS_LOCAL_VAD_PATH=""
                PARAMS_DOCKER_VAD_DIR=""
                if [ -z "$PARAMS_VAD_ID" ]; then
                    echo -e "    ${RED}The VAD model ID is empty, please setup again.${PLAIN}"
                else
                    break
                fi
            done
        elif [ "$PARAMS_VAD_ID" = "$LOCAL_MODEL" ]; then
            PARAMS_VAD_ID=""
            echo -e "    Please input VAD model path in local for FunASR server."

            while true
            do
                read -p "    Setting VAD model path in localhost: " PARAMS_LOCAL_VAD_PATH
                if [ -z "$PARAMS_LOCAL_VAD_PATH" ]; then
                    # use default vad model in Docker
                    echo -e "    ${RED}Please do not set an empty path in localhost.${PLAIN}"
                else
                    if [ ! -d "$PARAMS_LOCAL_VAD_PATH" ]; then
                        echo -e "    ${RED}The VAD model path set does not exist, please setup again.${PLAIN}"
                    else
                        # use vad model in localhost
                        PARAMS_LOCAL_VAD_DIR=$(dirname "$PARAMS_LOCAL_VAD_PATH")
                        vad_name=$(basename "$PARAMS_LOCAL_VAD_PATH")
                        PARAMS_DOCKER_VAD_DIR="/workspace/user_vad"
                        PARAMS_DOCKER_VAD_PATH=${PARAMS_DOCKER_VAD_DIR}/${vad_name}

                        echo -e "    ${UNDERLINE}You have chosen the model dir in localhost:${PLAIN} ${GREEN}${PARAMS_LOCAL_VAD_PATH}${PLAIN}"
                        echo -e "    ${UNDERLINE}The model dir in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_VAD_PATH}${PLAIN}"
                        echo
                        return 0
                    fi
                fi
            done
        fi

        PARAMS_DOCKER_VAD_DIR=$PARAMS_DOWNLOAD_MODEL_DIR
        PARAMS_DOCKER_VAD_PATH=${PARAMS_DOCKER_VAD_DIR}/${PARAMS_VAD_ID}

        echo -e "    ${UNDERLINE}The model ID is${PLAIN} ${GREEN}${PARAMS_VAD_ID}${PLAIN}"
        echo -e "    ${UNDERLINE}The model dir in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_VAD_PATH}${PLAIN}"
    fi

    echo
}

# Set punc model for FunASR server
setupPuncModelId(){
    echo -e "  ${UNDERLINE}${BOLD}[2.3/9]${PLAIN}"

    if [ -z "$PARAMS_DOWNLOAD_MODEL_DIR" ]; then
        # download_model_dir is empty, will use models in localhost.
        params_local_punc_path=`sed '/^PARAMS_LOCAL_PUNC_PATH=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
        if [ -z "$params_local_punc_path" ]; then
            PARAMS_LOCAL_PUNC_PATH=""
        else
            PARAMS_LOCAL_PUNC_PATH=${params_local_punc_path}
        fi

        echo -e "    ${YELLOW}Please input PUNC model path in local for FunASR server.${PLAIN}"
        echo -e "    Default: ${CYAN}${PARAMS_LOCAL_PUNC_PATH}${PLAIN}"

        while true
        do
            read -p "    Setting PUNC model path in localhost: " PARAMS_LOCAL_PUNC_PATH

            if [ -z "$PARAMS_LOCAL_PUNC_PATH" ]; then
                PARAMS_LOCAL_PUNC_PATH=${params_local_punc_path}
            fi
            if [ -z "$PARAMS_LOCAL_PUNC_PATH" ]; then
                # use default punc model in Docker
                PARAMS_LOCAL_PUNC_DIR=""
                PARAMS_DOCKER_PUNC_DIR=""
                PARAMS_DOCKER_PUNC_PATH="/workspace/models/punc"
                echo -e "    ${RED}Donnot set the local PUNC model path, will use PUNC model(${CYAN}/workspace/models/punc${PLAIN}${RED}) in Docker.${PLAIN}"

                echo -e "    ${UNDERLINE}You have chosen the default model dir in localhost: ${GREEN}${PARAMS_LOCAL_PUNC_PATH}${PLAIN}"
                echo -e "    ${UNDERLINE}The defalut model dir in Docker is ${GREEN}${PARAMS_DOCKER_PUNC_PATH}${PLAIN}"
                break
            else
                if [ ! -d "$PARAMS_LOCAL_PUNC_PATH" ]; then
                    echo -e "    ${RED}The PUNC model path set does not exist, please setup again.${PLAIN}"
                else
                    # use punc model in localhost
                    PARAMS_LOCAL_PUNC_DIR=$(dirname "$PARAMS_LOCAL_PUNC_PATH")
                    punc_name=$(basename "$PARAMS_LOCAL_PUNC_PATH")
                    PARAMS_DOCKER_PUNC_DIR="/workspace/user_punc"
                    PARAMS_DOCKER_PUNC_PATH=${PARAMS_DOCKER_PUNC_DIR}/${punc_name}

                    echo -e "    ${UNDERLINE}You have chosen the model dir in localhost:${PLAIN} ${GREEN}${PARAMS_LOCAL_PUNC_PATH}${PLAIN}"
                    echo -e "    ${UNDERLINE}The model dir in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_PUNC_PATH}${PLAIN}"
                    break
                fi
            fi
        done

        PARAMS_PUNC_ID=""
    else
        # please set model_id later.
        echo -e "    ${YELLOW}Please select PUNC model_id in ModelScope from the list below.${PLAIN}"

        menuSelection ${PUNC_MODELS[*]}
        result=$?
        index=`expr $result - 1`
        PARAMS_PUNC_ID=${PUNC_MODELS[${index}]}

        OTHERS="model_name"
        LOCAL_MODEL="model_path"
        if [ "$PARAMS_PUNC_ID" = "$OTHERS" ]; then
            params_punc_id=`sed '/^PARAMS_PUNC_ID=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
            if [ -z "$params_punc_id" ]; then
                PARAMS_PUNC_ID=""
            else
                PARAMS_PUNC_ID=${params_punc_id}
            fi

            echo -e "    Default: ${CYAN}${PARAMS_PUNC_ID}${PLAIN}"

            while true
            do
                read -p "    Setting PUNC model_id in ModelScope: " PARAMS_PUNC_ID

                PARAMS_LOCAL_PUNC_DIR=""
                PARAMS_LOCAL_PUNC_PATH=""
                PARAMS_DOCKER_PUNC_DIR=""
                if [ -z "$PARAMS_PUNC_ID" ]; then
                    echo -e "    ${RED}The PUNC model ID is empty, please setup again.${PLAIN}"
                else
                    break
                fi
            done
        elif [ "$PARAMS_PUNC_ID" = "$LOCAL_MODEL" ]; then
            PARAMS_PUNC_ID=""
            echo -e "    Please input PUNC model path in local for FunASR server."

            while true
            do
                read -p "    Setting PUNC model path in localhost: " PARAMS_LOCAL_PUNC_PATH
                if [ -z "$PARAMS_LOCAL_PUNC_PATH" ]; then
                    # use default punc model in Docker
                    echo -e "    ${RED}Please do not set an empty path in localhost.${PLAIN}"
                else
                    if [ ! -d "$PARAMS_LOCAL_PUNC_PATH" ]; then
                        echo -e "    ${RED}The PUNC model path set does not exist, please setup again.${PLAIN}"
                    else
                        # use punc model in localhost
                        PARAMS_LOCAL_PUNC_DIR=$(dirname "$PARAMS_LOCAL_PUNC_PATH")
                        punc_name=$(basename "$PARAMS_LOCAL_PUNC_PATH")
                        PARAMS_DOCKER_PUNC_DIR="/workspace/user_punc"
                        PARAMS_DOCKER_PUNC_PATH=${PARAMS_DOCKER_PUNC_DIR}/${punc_name}

                        echo -e "    ${UNDERLINE}You have chosen the model dir in localhost:${PLAIN} ${GREEN}${PARAMS_LOCAL_PUNC_PATH}${PLAIN}"
                        echo -e "    ${UNDERLINE}The model dir in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_PUNC_PATH}${PLAIN}"
                        echo
                        return 0
                    fi
                fi
            done
        fi

        PARAMS_DOCKER_PUNC_DIR=$PARAMS_DOWNLOAD_MODEL_DIR
        PARAMS_DOCKER_PUNC_PATH=${PARAMS_DOCKER_PUNC_DIR}/${PARAMS_PUNC_ID}

        echo -e "    ${UNDERLINE}The model ID is${PLAIN} ${GREEN}${PARAMS_PUNC_ID}${PLAIN}"
        echo -e "    ${UNDERLINE}The model dir in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_PUNC_PATH}${PLAIN}"
    fi

    echo
}

# Set server exec for FunASR
setupServerExec(){
    echo -e "${UNDERLINE}${BOLD}[3/9]${PLAIN}"

    params_docker_exec_path=`sed '/^PARAMS_DOCKER_EXEC_PATH=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    if [ -z "$params_docker_exec_path" ]; then
        PARAMS_DOCKER_EXEC_PATH="/workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server"
    else
        PARAMS_DOCKER_EXEC_PATH=${params_docker_exec_path}
    fi

    echo -e "  ${YELLOW}Please enter the path to the excutor of the FunASR service on the localhost.${PLAIN}"
    echo -e "  If not set, the default ${CYAN}/workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server${PLAIN} in Docker is used."
    read -p "  Setting the path to the excutor of the FunASR service on the localhost: " PARAMS_LOCAL_EXEC_PATH

    if [ -z "$PARAMS_LOCAL_EXEC_PATH" ]; then
        PARAMS_DOCKER_EXEC_PATH="/workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server"
    else
        if [ ! -d "$PARAMS_LOCAL_EXEC_PATH" ]; then
            echo -e "  ${RED}The FunASR server path set does not exist, will use default.${PLAIN}"
            PARAMS_LOCAL_EXEC_PATH=""
            PARAMS_LOCAL_EXEC_DIR=""
            PARAMS_DOCKER_EXEC_PATH="/workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server"
            PARAMS_DOCKER_EXEC_DIR="/workspace/FunASR/funasr/runtime/websocket/build/bin"
        else
            PARAMS_LOCAL_EXEC_DIR=$(dirname "$PARAMS_LOCAL_EXEC_PATH")
            exec=$(basename "$PARAMS_LOCAL_EXEC_PATH")
            PARAMS_DOCKER_EXEC_DIR="/server"
            PARAMS_DOCKER_EXEC_PATH=${PARAMS_DOCKER_EXEC_DIR}/${exec}
            echo -e "  ${UNDERLINE}The path of FunASR in localhost is${PLAIN} ${GREEN}${PARAMS_LOCAL_EXEC_PATH}${PLAIN}"
        fi
    fi
    echo -e "  ${UNDERLINE}Corresponding, the path of FunASR in Docker is${PLAIN} ${GREEN}${PARAMS_DOCKER_EXEC_PATH}${PLAIN}"

    echo
}

# Configure FunASR server host port setting
setupHostPort(){
    echo -e "${UNDERLINE}${BOLD}[4/9]${PLAIN}"

    params_host_port=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    if [ -z "$params_host_port" ]; then
        PARAMS_HOST_PORT="10095"
    else
        PARAMS_HOST_PORT=${params_host_port}
    fi

    while true
    do
        echo -e "  ${YELLOW}Please input the opened port in the host used for FunASR server.${PLAIN}"
        echo -e "  Default: ${CYAN}${PARAMS_HOST_PORT}${PLAIN}"
        read -p "  Setting the opened host port [1-65535]: " PARAMS_HOST_PORT

        if [ -z "$PARAMS_HOST_PORT" ]; then
            params_host_port=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
            if [ -z "$params_host_port" ]; then
                PARAMS_HOST_PORT="10095"
            else
                PARAMS_HOST_PORT=${params_host_port}
            fi
        fi
        expr ${PARAMS_HOST_PORT} + 0 &>/dev/null
        if [ $? -eq 0 ]; then
            if [ ${PARAMS_HOST_PORT} -ge 1 ] && [ ${PARAMS_HOST_PORT} -le 65535 ]; then
                echo -e "  ${UNDERLINE}The port of the host is${PLAIN} ${GREEN}${PARAMS_HOST_PORT}${PLAIN}"
                echo -e "  ${UNDERLINE}The port in Docker for FunASR server is ${PLAIN}${GREEN}${PARAMS_DOCKER_PORT}${PLAIN}"
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

setupThreadNum(){
    echo -e "${UNDERLINE}${BOLD}[5/9]${PLAIN}"

    params_decoder_thread_num=`sed '/^PARAMS_DECODER_THREAD_NUM=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    if [ -z "$params_decoder_thread_num" ]; then
        PARAMS_DECODER_THREAD_NUM=$CPUNUM
    else
        PARAMS_DECODER_THREAD_NUM=${params_decoder_thread_num}
    fi

    while true
    do
        echo -e "  ${YELLOW}Please input thread number for FunASR decoder.${PLAIN}"
        echo -e "  Default: ${CYAN}${PARAMS_DECODER_THREAD_NUM}${PLAIN}"
        read -p "  Setting the number of decoder thread: " PARAMS_DECODER_THREAD_NUM

        if [ -z "$PARAMS_DECODER_THREAD_NUM" ]; then
            params_decoder_thread_num=`sed '/^PARAMS_DECODER_THREAD_NUM=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
            if [ -z "$params_decoder_thread_num" ]; then
                PARAMS_DECODER_THREAD_NUM=$CPUNUM
            else
                PARAMS_DECODER_THREAD_NUM=${params_decoder_thread_num}
            fi
        fi
        expr ${PARAMS_DECODER_THREAD_NUM} + 0 &>/dev/null
        if [ $? -eq 0 ]; then
            if [ ${PARAMS_DECODER_THREAD_NUM} -ge 1 ] && [ ${PARAMS_DECODER_THREAD_NUM} -le 65535 ]; then
                break
            else
                echo -e "  ${RED}Input error, please input correct number!${PLAIN}"
            fi
        else
            echo -e "  ${RED}Input error, please input correct number!${PLAIN}"
        fi
        done
    echo

    multiple_io=4
    PARAMS_DECODER_THREAD_NUM=`expr $PARAMS_DECODER_THREAD_NUM + 0`
    PARAMS_IO_THREAD_NUM=`expr $PARAMS_DECODER_THREAD_NUM / $multiple_io`
    if [ $PARAMS_IO_THREAD_NUM -eq 0 ]; then
        PARAMS_IO_THREAD_NUM=1
    fi

    echo -e "  ${UNDERLINE}The number of decoder threads is${PLAIN} ${GREEN}${PARAMS_DECODER_THREAD_NUM}${PLAIN}"
    echo -e "  ${UNDERLINE}The number of IO threads is${PLAIN} ${GREEN}${PARAMS_IO_THREAD_NUM}${PLAIN}"
    echo
}

paramsFromDefault(){
    echo -e "${UNDERLINE}${BOLD}[2-5/9]${PLAIN}"
    echo -e "  ${YELLOW}Load parameters from ${FUNASR_CONFIG_FILE}${PLAIN}"
    echo

    PARAMS_DOCKER_IMAGE=`sed '/^PARAMS_DOCKER_IMAGE=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_LOCAL_MODELS_DIR=`sed '/^PARAMS_LOCAL_MODELS_DIR=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_DOWNLOAD_MODEL_DIR=`sed '/^PARAMS_DOWNLOAD_MODEL_DIR=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_LOCAL_ASR_PATH=`sed '/^PARAMS_LOCAL_ASR_PATH=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_DOCKER_ASR_PATH=`sed '/^PARAMS_DOCKER_ASR_PATH=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_ASR_ID=`sed '/^PARAMS_ASR_ID=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_LOCAL_VAD_PATH=`sed '/^PARAMS_LOCAL_VAD_PATH=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_DOCKER_VAD_PATH=`sed '/^PARAMS_DOCKER_VAD_PATH=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_VAD_ID=`sed '/^PARAMS_VAD_ID=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_LOCAL_PUNC_PATH=`sed '/^PARAMS_LOCAL_PUNC_PATH=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_DOCKER_PUNC_PATH=`sed '/^PARAMS_DOCKER_PUNC_PATH=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_PUNC_ID=`sed '/^PARAMS_PUNC_ID=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_DOCKER_EXEC_PATH=`sed '/^PARAMS_DOCKER_EXEC_PATH=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_HOST_PORT=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_DOCKER_PORT=`sed '/^PARAMS_DOCKER_PORT=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_DECODER_THREAD_NUM=`sed '/^PARAMS_DECODER_THREAD_NUM=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
    PARAMS_IO_THREAD_NUM=`sed '/^PARAMS_IO_THREAD_NUM=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
}

saveParams(){
    echo "$i" > $FUNASR_CONFIG_FILE
    echo -e "  ${GREEN}Parameters are stored in the file ${FUNASR_CONFIG_FILE}${PLAIN}"

    echo "PARAMS_DOCKER_IMAGE=${PARAMS_DOCKER_IMAGE}" > $FUNASR_CONFIG_FILE
    echo "PARAMS_LOCAL_MODELS_DIR=${PARAMS_LOCAL_MODELS_DIR}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_DOWNLOAD_MODEL_DIR=${PARAMS_DOWNLOAD_MODEL_DIR}" >> $FUNASR_CONFIG_FILE

    echo "PARAMS_LOCAL_EXEC_PATH=${PARAMS_LOCAL_EXEC_PATH}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_LOCAL_EXEC_DIR=${PARAMS_LOCAL_EXEC_DIR}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_EXEC_PATH=${PARAMS_DOCKER_EXEC_PATH}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_EXEC_DIR=${PARAMS_DOCKER_EXEC_DIR}" >> $FUNASR_CONFIG_FILE

    echo "PARAMS_LOCAL_ASR_PATH=${PARAMS_LOCAL_ASR_PATH}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_LOCAL_ASR_DIR=${PARAMS_LOCAL_ASR_DIR}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_ASR_PATH=${PARAMS_DOCKER_ASR_PATH}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_ASR_DIR=${PARAMS_DOCKER_ASR_DIR}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_ASR_ID=${PARAMS_ASR_ID}" >> $FUNASR_CONFIG_FILE

    echo "PARAMS_LOCAL_PUNC_PATH=${PARAMS_LOCAL_PUNC_PATH}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_LOCAL_PUNC_DIR=${PARAMS_LOCAL_PUNC_DIR}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_PUNC_PATH=${PARAMS_DOCKER_PUNC_PATH}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_PUNC_DIR=${PARAMS_DOCKER_PUNC_DIR}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_PUNC_ID=${PARAMS_PUNC_ID}" >> $FUNASR_CONFIG_FILE

    echo "PARAMS_LOCAL_VAD_PATH=${PARAMS_LOCAL_VAD_PATH}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_LOCAL_VAD_DIR=${PARAMS_LOCAL_VAD_DIR}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_VAD_PATH=${PARAMS_DOCKER_VAD_PATH}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_VAD_DIR=${PARAMS_DOCKER_VAD_DIR}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_VAD_ID=${PARAMS_VAD_ID}" >> $FUNASR_CONFIG_FILE

    echo "PARAMS_HOST_PORT=${PARAMS_HOST_PORT}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_DOCKER_PORT=${PARAMS_DOCKER_PORT}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_DECODER_THREAD_NUM=${PARAMS_DECODER_THREAD_NUM}" >> $FUNASR_CONFIG_FILE
    echo "PARAMS_IO_THREAD_NUM=${PARAMS_IO_THREAD_NUM}" >> $FUNASR_CONFIG_FILE
}

showAllParams(){
    echo -e "${UNDERLINE}${BOLD}[6/9]${PLAIN}"
    echo -e "  ${YELLOW}Show parameters of FunASR server setting and confirm to run ...${PLAIN}"
    echo

    if [ ! -z "$PARAMS_DOCKER_IMAGE" ]; then
        echo -e "  The current Docker image is                                    : ${GREEN}${PARAMS_DOCKER_IMAGE}${PLAIN}"
    fi

    if [ ! -z "$PARAMS_LOCAL_MODELS_DIR" ]; then
        echo -e "  The model is downloaded or stored to this directory in local   : ${GREEN}${PARAMS_LOCAL_MODELS_DIR}${PLAIN}"
    fi
    if [ ! -z "$PARAMS_DOWNLOAD_MODEL_DIR" ]; then
        echo -e "  The model will be automatically downloaded to the directory    : ${GREEN}${PARAMS_DOWNLOAD_MODEL_DIR}${PLAIN}"
    fi

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

    if [ ! -z "$PARAMS_LOCAL_EXEC_PATH" ]; then
        echo -e "  The local path of the FunASR service executor                 : ${GREEN}${PARAMS_LOCAL_EXEC_PATH}${PLAIN}"
    fi
    echo -e "  The path in the docker of the FunASR service executor          : ${GREEN}${PARAMS_DOCKER_EXEC_PATH}${PLAIN}"

    echo -e "  Set the host port used for use by the FunASR service           : ${GREEN}${PARAMS_HOST_PORT}${PLAIN}"
    echo -e "  Set the docker port used by the FunASR service                 : ${GREEN}${PARAMS_DOCKER_PORT}${PLAIN}"

    echo -e "  Set the number of threads used for decoding the FunASR service : ${GREEN}${PARAMS_DECODER_THREAD_NUM}${PLAIN}"
    echo -e "  Set the number of threads used for IO the FunASR service       : ${GREEN}${PARAMS_IO_THREAD_NUM}${PLAIN}"

    echo
    while true
    do
        params_confirm="y"
        echo -e "  ${YELLOW}Please input [Y/n] to confirm the parameters.${PLAIN}"
        echo -e "  [y] Verify that these parameters are correct and that the service will run."
        echo -e "  [n] The parameters set are incorrect, it will be rolled out, please rerun."
        read -p "  read confirmation[Y/n]: " params_confirm

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
installDocker(){
    echo -e "${UNDERLINE}${BOLD}[7/9]${PLAIN}"

    if [ $DOCKERINFOLEN -gt 30 ]; then
        echo -e "  ${YELLOW}Docker has installed.${PLAIN}"
    else
        lowercase_osid=$(echo $OSID | tr '[A-Z]' '[a-z]')
        echo -e "  ${YELLOW}Start install docker for $lowercase_osid ${PLAIN}"
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
            *)
                echo "$lowercase_osid is not supported."
                ;;
        esac

        echo -e "  Get docker installer: ${GREEN}$DOCKER_INSTALL_CMD${PLAIN}"
        echo -e "  Get docker run: ${GREEN}$DOCKER_INSTALL_RUN_CMD${PLAIN}"

        $DOCKER_INSTALL_CMD
        if [ ! -z "$DOCKER_INSTALL_RUN_CMD" ]; then
            $DOCKER_INSTALL_RUN_CMD
        fi

        DOCKERINFO=$(sudo docker info | wc -l)
        DOCKERINFOLEN=$(expr $DOCKERINFO)
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
}

# Download docker image
downloadDockerImage(){
    echo -e "${UNDERLINE}${BOLD}[8/9]${PLAIN}"
    echo -e "  ${YELLOW}Pull docker image(${PARAMS_DOCKER_IMAGE})...${PLAIN}"

    sudo docker pull ${PARAMS_DOCKER_IMAGE}

    echo
    sleep 1
}

dockerRun(){
    echo -e "${UNDERLINE}${BOLD}[9/9]${PLAIN}"
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
    if [ ! -z "$PARAMS_LOCAL_EXEC_DIR" ]; then
        if [ -z "$DIR_MAP_PARAMS" ]; then
            DIR_MAP_PARAMS="${DIR_PARAMS} -v ${PARAMS_LOCAL_EXEC_DIR}:${PARAMS_DOCKER_EXEC_DIR}"
        else
            DIR_MAP_PARAMS="${DIR_MAP_PARAMS} -v ${PARAMS_LOCAL_EXEC_DIR}:${PARAMS_DOCKER_EXEC_DIR}"
        fi
    fi
    if [ ! -z "$PARAMS_LOCAL_MODELS_DIR" ]; then
        if [ -z "$DIR_MAP_PARAMS" ]; then
            DIR_MAP_PARAMS="${DIR_PARAMS} -v ${PARAMS_LOCAL_MODELS_DIR}:${PARAMS_DOWNLOAD_MODEL_DIR}"
        else
            DIR_MAP_PARAMS="${DIR_MAP_PARAMS} -v ${PARAMS_LOCAL_MODELS_DIR}:${PARAMS_DOWNLOAD_MODEL_DIR}"
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
    result=`expr $result + 0`
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
        stage=`expr $result + 0`
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
    echo -e "  ${BOLD}If you want to see an example of how to use the client, you can run ${PLAIN}${GREEN}sudo bash funasr-runtime-deploy.sh -c${PLAIN} ."
    echo
}

checkDockerExist(){
    result=$(sudo docker ps | grep ${PARAMS_DOCKER_IMAGE} | wc -l)
    result=`expr $result + 0`
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
    model_id=$1

    result=$(echo $1 | grep "asr")
    if [[ "$result" != "" ]]
    then
        PARAMS_ASR_ID=$1
        PARAMS_DOCKER_ASR_PATH=${PARAMS_DOCKER_ASR_DIR}/${PARAMS_ASR_ID}
        return 0
    fi
    result=$(echo $1 | grep "vad")
    if [[ "$result" != "" ]]
    then
        PARAMS_VAD_ID=$1
        PARAMS_DOCKER_VAD_PATH=${PARAMS_DOCKER_VAD_DIR}/${PARAMS_VAD_ID}
        retun 0
    fi
    result=$(echo $1 | grep "punc")
    if [[ "$result" != "" ]]
    then
        PARAMS_PUNC_ID=$1
        PARAMS_DOCKER_PUNC_PATH=${PARAMS_DOCKER_PUNC_DIR}/${PARAMS_PUNC_ID}
        retun 0
    fi
}

sampleClientRun(){
    echo -e "${YELLOW}Will download sample tools for the client to show how speech recognition works.${PLAIN}"

    sample_name="funasr_samples"
    sample_tar="funasr_samples.tar.gz"
    sample_url="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/${sample_tar}"
    DOWNLOAD_SAMPLE="curl -O ${sample_url}"
    UNTAR_CMD="tar -zxf ${sample_tar}"

    if [ ! -f "${sample_tar}" ]; then
        ${DOWNLOAD_SAMPLE}
    fi
    if [ -f "${sample_tar}" ]; then
        ${UNTAR_CMD}
    fi
    if [ -d "${sample_name}" ]; then

        echo -e "  Please select the client you want to run."
        menuSelection ${SAMPLE_CLIENTS[*]}
        result=$?
        index=`expr $result - 1`
        lang=${SAMPLE_CLIENTS[${index}]}
        echo

        SERVER_IP="127.0.0.1"
        read -p "  Please enter the IP of server, default(${SERVER_IP}): " SERVER_IP
        if [ -z "$SERVER_IP" ]; then
            SERVER_IP="127.0.0.1"
        fi

        HOST_PORT=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
        if [ -z "$HOST_PORT" ]; then
            HOST_PORT="10095"
        fi
        read -p "  Please enter the port of server, default(${HOST_PORT}): " HOST_PORT
        if [ -z "$HOST_PORT" ]; then
            HOST_PORT=`sed '/^PARAMS_HOST_PORT=/!d;s/.*=//' ${FUNASR_CONFIG_FILE}`
            if [ -z "$HOST_PORT" ]; then
                HOST_PORT="10095"
            fi
        fi

        WAV_PATH="${cur_dir}/funasr_samples/audio/asr_example.wav"
        read -p "  Please enter the audio path, default(${WAV_PATH}): " WAV_PATH
        if [ -z "$WAV_PATH" ]; then
            WAV_PATH="${cur_dir}/funasr_samples/audio/asr_example.wav"
        fi

        echo
        PRE_CMD=”“
        case "$lang" in
            Linux_Cpp)
                PRE_CMD="export LD_LIBRARY_PATH=${cur_dir}/funasr_samples/cpp/libs:\$LD_LIBRARY_PATH"
                CLIENT_EXEC="${cur_dir}/funasr_samples/cpp/funasr-wss-client"
                RUN_CMD="${CLIENT_EXEC} --server-ip ${SERVER_IP} --port ${HOST_PORT} --wav-path ${WAV_PATH}"
                echo -e "  Run ${BLUE}${PRE_CMD}${PLAIN}"
                ${PRE_CMD}
                echo
                ;;
            Python)
                CLIENT_EXEC="${cur_dir}/funasr_samples/python/wss_client_asr.py"
                RUN_CMD="python3 ${CLIENT_EXEC} --host ${SERVER_IP} --port ${HOST_PORT} --mode offline --audio_in ${WAV_PATH} --send_without_sleep --output_dir ./funasr_samples/python"
                PRE_CMD="pip3 install click>=8.0.4"
                echo -e "  Run ${BLUE}${PRE_CMD}${PLAIN}"
                ${PRE_CMD}
                echo
                PRE_CMD="pip3 install -r ${cur_dir}/funasr_samples/python/requirements_client.txt"
                echo -e "  Run ${BLUE}${PRE_CMD}${PLAIN}"
                ${PRE_CMD}
                echo
                ;;
            *)
                echo "$lang is not supported."
                ;;
        esac

        echo -e "  Run ${BLUE}${RUN_CMD}${PLAIN}"
        ${RUN_CMD}
        echo
        echo -e "  If failed, you can try (${GREEN}${RUN_CMD}${PLAIN}) in your Shell."
        echo
    fi
}

# Install main function
installFunasrDocker(){
    installDocker
    downloadDockerImage
}

modelsConfigure(){
    setupModelType
    setupAsrModelId
    setupVadModelId
    setupPuncModelId
}

paramsConfigure(){
    selectDockerImages
    result=$?
    result=`expr $result + 0`
    if [ ${result} -eq 50 ]; then
        return 50
    fi

    setupModelType
    setupAsrModelId
    setupVadModelId
    setupPuncModelId
    setupServerExec
    setupHostPort
    setupThreadNum
}

# Display Help info
displayHelp(){
    echo -e "${UNDERLINE}Usage${PLAIN}:"
    echo -e "  $0 [OPTIONAL FLAGS]"
    echo
    echo -e "funasr-runtime-deploy.sh - a Bash script to install&run FunASR docker."
    echo
    echo -e "${UNDERLINE}Options${PLAIN}:"
    echo -e "   ${BOLD}-i, --install${PLAIN}      Install and run FunASR docker."
    echo -e "   ${BOLD}-s, --start${PLAIN}        Run FunASR docker with configuration that has already been set."
    echo -e "   ${BOLD}-p, --stop${PLAIN}         Stop FunASR docker."
    echo -e "   ${BOLD}-r, --restart${PLAIN}      Restart FunASR docker."
    echo -e "   ${BOLD}-u, --update${PLAIN}       Update the model ID that has already been set, e.g: --update model XXXX."
    echo -e "   ${BOLD}-c, --client${PLAIN}       Get a client example to show how to initiate speech recognition."
    echo -e "   ${BOLD}-v, --version${PLAIN}      Display current script version."
    echo -e "   ${BOLD}-h, --help${PLAIN}         Display this help."
    echo
    echo -e "${UNDERLINE}funasr-runtime-deploy.sh${PLAIN} - Version ${scriptVersion} "
    echo -e "Modify Date ${scriptDate}"
}

# OS
OSID=$(grep ^ID= /etc/os-release | cut -d= -f2)
OSVER=$(lsb_release -cs)
OSNUM=$(grep -oE  "[0-9.]+" /etc/issue)
CPUNUM=$(cat /proc/cpuinfo |grep "processor"|wc -l)
DOCKERINFO=$(sudo docker info | wc -l)
DOCKERINFOLEN=$(expr $DOCKERINFO)

# PARAMS
FUNASR_CONFIG_FILE="/var/funasr/config"
#  The path of server executor in local
PARAMS_LOCAL_EXEC_PATH=""
#  The dir stored server excutor in local
PARAMS_LOCAL_EXEC_DIR=""
#  The server excutor in local
PARAMS_DOCKER_EXEC_PATH="/workspace/FunASR/funasr/runtime/websocket/build/bin/funasr-wss-server"
#  The dir stored server excutor in docker
PARAMS_DOCKER_EXEC_DIR="/workspace/FunASR/funasr/runtime/websocket/build/bin"

#  The dir of model in local
PARAMS_LOCAL_MODELS_DIR=""
#  The dir for downloading model in docker
PARAMS_DOWNLOAD_MODEL_DIR=""
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
PARAMS_PUNC_ID=""

#  The dir stored vad model in local
PARAMS_LOCAL_VAD_DIR=""
#  The path of vad model in local
PARAMS_LOCAL_VAD_PATH=""
#  The dir stored vad model in docker
PARAMS_DOCKER_VAD_DIR=""
#  The path of vad model in docker
PARAMS_DOCKER_VAD_PATH=""
#  The vad model ID in ModelScope
PARAMS_VAD_ID=""

#  The dir stored asr model in local
PARAMS_LOCAL_ASR_DIR=""
#  The path of asr model in local
PARAMS_LOCAL_ASR_PATH=""
#  The dir stored asr model in docker
PARAMS_DOCKER_ASR_DIR=""
#  The path of asr model in docker
PARAMS_DOCKER_ASR_PATH=""
#  The asr model ID in ModelScope
PARAMS_ASR_ID=""

PARAMS_HOST_PORT="10095"
PARAMS_DOCKER_PORT="10095"
PARAMS_DECODER_THREAD_NUM="32"
PARAMS_IO_THREAD_NUM="8"


echo -e "#############################################################"
echo -e "#       ${RED}OS${PLAIN}: $OSID $OSNUM $OSVER "
echo -e "#   ${RED}Kernel${PLAIN}: $(uname -m) Linux $(uname -r)"
echo -e "#      ${RED}CPU${PLAIN}: $(grep 'model name' /proc/cpuinfo | uniq | awk -F : '{print $2}' | sed 's/^[ \t]*//g' | sed 's/ \+/ /g') "
echo -e "#  ${RED}CPU NUM${PLAIN}: $CPUNUM"
echo -e "#      ${RED}RAM${PLAIN}: $(cat /proc/meminfo | grep 'MemTotal' | awk -F : '{print $2}' | sed 's/^[ \t]*//g') "
echo -e "#############################################################"
echo

# Initialization step
case "$1" in
    install|-i|--install)
        rootNess
        paramsConfigure
        result=$?
        result=`expr $result + 0`
        if [ ${result} -ne 50 ]; then
            showAllParams
            installFunasrDocker
            dockerRun
            result=$?
            stage=`expr $result + 0`
            if [ ${stage} -eq 98 ]; then
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
        stage=`expr $result + 0`
        if [ ${stage} -eq 98 ]; then
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
        stage=`expr $result + 0`
        if [ ${stage} -eq 98 ]; then
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

        if [ $# -eq 1 ];
        then
            modelsConfigure
        elif [ $# -eq 3 ];
        then
            type=$2
            id=$3
            MODEL="model"
            if [ "$type" = "$MODEL" ]; then
                modelChange $id
            else
                modelsConfigure
            fi
        else
            modelsConfigure
        fi

        saveParams
        dockerExit
        dockerRun
        result=$?
        stage=`expr $result + 0`
        if [ ${stage} -eq 98 ]; then
            dockerExit
            dockerRun
        fi
        ;;
    client|-c|--client)
        rootNess
        sampleClientRun
        ;;
    *)
        clear
        displayHelp
        exit 0
        ;;
esac