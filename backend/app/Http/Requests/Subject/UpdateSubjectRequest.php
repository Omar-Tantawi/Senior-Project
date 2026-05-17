<?php

namespace App\Http\Requests\Subject;

use Illuminate\Foundation\Http\FormRequest;

class UpdateSubjectRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'name' => 'sometimes|string|max:100',
            'code' => 'sometimes|string|max:20|unique:subjects,code,' . $this->route('id'),
        ];
    }
}
