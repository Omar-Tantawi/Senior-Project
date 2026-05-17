<?php

namespace App\Http\Requests\Message;

use Illuminate\Foundation\Http\FormRequest;

class SendMessageRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'teacher_id' => 'required|exists:teachers,id',
            'student_id' => 'nullable|integer',
            'subject'    => 'required|string|max:255',
            'body'       => 'required|string',
        ];
    }
}
