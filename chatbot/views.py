from django.shortcuts import redirect
from django.template import loader
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from .functions import ChatBotModel

# Create your views here.
def home(request):
    template = loader.get_template('chatbot.html')
    context = {}
    return HttpResponse(template.render(context, request))

def handleInput(request):
    if request.method == 'POST':
        user_message = request.POST.get('user_message')
        
        chatbot = ChatBotModel()
        bot_reply = chatbot.predict_response(user_message)
        
        # Return JSON response
        return JsonResponse({'bot_reply': bot_reply})
    
    return HttpResponse(status=405)  # Method not allowed