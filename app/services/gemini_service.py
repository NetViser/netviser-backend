
from fastapi import HTTPException
from app.configs.config import get_settings
from google import genai
from google.genai import types

settings = get_settings()

class GeminiService:
    """
    Singleton class for interacting with the Gemini 2.0 Flash API,
    specifically for generating explanations such as SHAP value interpretations.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiService, cls).__new__(cls)
            cls._instance.initialize_client()
        return cls._instance

    def initialize_client(self):
        """
        Initialize the Gemini API client using the API key from settings.
        """
        try:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing Gemini client: {str(e)}")

    def generate_shap_force_plot_explanation(self, input_shap_force_plot_data: str, max_tokens: int = 1000) -> str:
        """
        Generate a text explanation for the provided SHAP force plot data using Gemini 2.0 Flash.
        This method uses a predefined few-shot prompt template to guide the explanation generation.
        
        :param input_shap_force_plot_data: The SHAP force plot data to be explained.
        :param max_tokens: Maximum number of tokens for the output text (default: 1000).
        :return: The generated explanation text.
        :raises HTTPException: If the Gemini API call fails.
        """
        few_shot_prompt = f"""
**System/Instruction:**  
You are an AI assistant with expertise in explaining machine learning model predictions. You will be given SHAP force plot data for a single prediction made by a model that classifies an instance into potential network attacks (e.g., FTP-Patator, DDoS, PortScan). Your task is to provide a concise, beginner-friendly explanation in **one paragraph**. In your explanation:
1. Briefly explain what the classified attack is (its name and typical characteristics).  
2. Describe what a SHAP force plot is, including the significance of the base (expected) value.  
3. Explain how each feature’s value and its SHAP value pull the prediction from the base value toward or away from the final classification.  

Follow the style and detail shown in **Example 1** below. Then apply it to the **New Input**.

---

**Example 1**  
**Input (SHAP Force Plot Data):**  
```
Base (Expected) Value: -5.350231170654297

Feature                  Feature Value    SHAP Value
Dst Port                21               10.3567
Bwd IAT Mean            795787.2         0.3808
Total TCP Flow Time     4018946          0.3753
Src Port                52148            0.2012
Total Length of Fwd Packet   30         0.0487
Fwd Packet Length Max   16               0.0091
Bwd Packet Length Std   14.68            0.0064
Bwd Init Win Bytes      227              0.0000
Flow IAT Min            3                0.0000
Fwd PSH Flags           2                0.0000
```

**Output (Single Paragraph Explanation):**  
```
FTP-Patator is a brute force attack targeting FTP servers where attackers repeatedly try different username and password combinations to gain unauthorized access, often generating a high volume of login attempts that stand out in network traffic. In the given SHAP force plot, the model starts with a base value of -5.350231170654297, which represents the expected output before feature contributions are applied. The 'Dst Port' feature, with a value of 21 and a SHAP value of 10.3567, significantly pushes the prediction toward FTP-Patator, likely because port 21 is standard for FTP connections, while features like 'Bwd IAT Mean' (795787.2, SHAP 0.3808) and 'Total TCP Flow Time' (4018946, SHAP 0.3753) provide smaller positive contributions, and other features such as 'Src Port' (52148, SHAP 0.2012) and 'Total Length of Fwd Packet' (30, SHAP 0.0487) add minimal influence; the remaining features have negligible impact.
```

---

**New Input (SHAP Force Plot Data):**  
```
{input_shap_force_plot_data}
```

**Your Task:**  
Provide a single-paragraph explanation in the same style as **Example 1**, using the **New Input** data.
"""
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,  # e.g., "gemini-2.0-flash"
                contents=[few_shot_prompt],
                config=types.GenerateContentConfig(max_output_tokens=max_tokens)
            )
            return response.text
        except Exception as e:
            print(f"Error generating SHAP force plot explanation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
