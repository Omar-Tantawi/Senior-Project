<?php

namespace App\Http\Requests\Parent;

use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Validation\Rules\Password;

class StoreParentRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'name'                    => 'required|string|max:255',
            'email'                   => 'required|email|unique:users,email',
            'phone'                   => 'nullable|string|max:20',
            'password'                => ['required', Password::min(8)],
            'children'                => 'sometimes|array',
            'children.*.student_id'   => 'required_with:children|exists:students,id',
            'children.*.relationship' => 'required_with:children|string|max:50',
            'children.*.isprimary'    => 'sometimes|boolean',
        ];
    }
}
