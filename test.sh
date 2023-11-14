curl --location 'http://121.127.44.53:8102/v1/chat/completions' \
--header 'Authorization: Bearer 8743c0de-6614-456a-a1ac-2f0d5069823d' \
--header 'Content-Type: application/json' \
--data "{
    \"model\": \"gpt-4\",
    \"messages\": [
        {
            \"role\": \"user\",
            \"content\": \"1+1=？'\"
        }
    ],
}"