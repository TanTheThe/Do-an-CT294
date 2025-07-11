import axios from 'axios';

const apiUrl = 'https://abalone-api.onrender.com'

export const postDataApi = async (url, data) => {
    try {
        const response = await axios.post(
            apiUrl + url,
            data
            )

        return {
            success: true,
            statusCode: 200,
            data: response.data.content,
            message: response.data.message
        }

    } catch (error) {
        return {
            success: false,
            statusCode: 400,
            data: error?.response?.data
        }
    }
}