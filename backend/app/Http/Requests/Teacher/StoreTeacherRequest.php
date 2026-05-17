<?php

namespace App\Http\Requests\Teacher;

use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Validation\Rules\Password;

class StoreTeacherRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'name'          => 'required|string|max:255',
            'email'         => 'required|email|unique:users,email',
            'phone'         => 'nullable|string|max:20',
            'password'      => ['required', Password::min(8)],
            'date_of_birth' => 'nullable|date',
            'gender'        => 'nullable|in:male,female,other',
            'address'       => 'nullable|string',
            'hire_date'     => 'nullable|date',
            'status'        => 'nullable|in:active,inactive,resigned',
        ];
    }
}
