<?php

namespace App\Http\Requests\Driver;

use App\Models\Driver;
use Illuminate\Foundation\Http\FormRequest;

class UpdateDriverRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        $userId = Driver::findOrFail($this->route('id'))->user_id;

        return [
            'name'      => 'sometimes|string|max:255',
            'email'     => "sometimes|email|unique:users,email,{$userId}",
            'phone'     => 'nullable|string|max:20',
            'is_active' => 'nullable|boolean',
        ];
    }
}
